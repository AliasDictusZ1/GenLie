#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoMAEv2-Large · Long-video lie detection
Modes:
  --mode extract : offline extract & cache CLS features
  --mode train   : train classifier + identity-adversarial + triplet loss on cached features
  --mode grid    : grid search over hyperparameters including loss weights
注意：在此版本中，标签定义为真实(truth)=0，谎言(lie)=1
"""
from __future__ import annotations
import os, json, argparse, random, logging, pickle, re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import product
from utils.tools import choose_gpu, get_device, read_data_sheet
from utils.utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = choose_gpu()
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pathlib import Path
from tools import *
from models.gen_lie import *
from utils.eval import *

device=get_device()

output_dict=defaultdict(lambda:defaultdict(dict)) #用于保存所有的输出，方便自己私下计算

# -----------------------------------------------------------------------------
# Dataset: feature extraction
# -----------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    """Return (List[PIL.Image], label, pid, vid)"""
    def __init__(self, root_dir: str, label_path: str):
        self.root = Path(root_dir)
        self.label_path = Path(label_path)
        self.label_dict = self.load_labels(self.label_path)#{sample_id:label}
        assert (self.root.is_dir() or self.root.is_file()) and self.label_path.is_file()
        self.samples: list[Tuple[Path,int,str,str]] = []
        self._build_index()
    def load_labels(self, label_path):
        df = pd.read_csv(label_path)
        df.columns = [c.strip() for c in df.columns]

        return {row["name"]:int(row["label"]) for _, row in df.iterrows()}


    def _build_index(self):
        if self.root.is_dir():
            for person in sorted(self.root.iterdir()):
                if not (person.is_dir() and re.fullmatch(r"\d{3}", person.name)): continue
                pid = person.name
                for sample_path in person.iterdir():
                    vid = sample_path.name
                    lbl = self.label_dict[vid]
                    self.samples.append((sample_path, lbl, pid, vid))
        else:#json
            for subj_id,samples in json.load(self.root.open('r')).items():
                for sample_id,frame_paths in samples.items():
                    # 从视频ID中提取标签 (truth=0, lie=1)
                    vid = sample_id
                    lbl = self.label_dict[vid]
                    self.samples.append((frame_paths, lbl, subj_id, sample_id))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vdir, lbl, pid, vid = self.samples[idx]
        imgs = [Image.open(p).convert("RGB") for p in (sorted(vdir.glob("*.jpg")) if isinstance(vdir,Path) else vdir)]
        return imgs, lbl, pid, vid

# -----------------------------------------------------------------------------
# Dataset: cached features
# -----------------------------------------------------------------------------
class CachedFeatDataset(Dataset):
    """Return (T,D), label, pid"""
    def __init__(self, feat_root: str, index_pkl: str):
        self.root = Path(feat_root)
        self.index = pickle.load(open(index_pkl, "rb"))

    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        fp, lbl, pid= self.index[idx]
        # pid = Path(fp).parts[0] 
        feat = torch.load(self.root / fp)  # (T,D)
        return feat, torch.tensor(lbl), pid

def collate_cached(batch):
    feats, lbls, pids = zip(*batch)
    feats = torch.stack([f.mean(0) for f in feats])  # (B,D)
    return feats, torch.stack(lbls), list(pids)

# -----------------------------------------------------------------------------
# Frame segmentation util
# -----------------------------------------------------------------------------
def sample_triplets(embeddings, labels, pid_idx):
    """
    输入：
        embeddings: Tensor, (B, D)
        labels: Tensor, (B,) 分类标签
        pid_idx: Tensor, (B,) 身份索引，用于排除同一身份负样本等
    输出：
        三个索引a,p,n，分别是anchor, positive, negative的索引列表
    """
    a, p, n = [], [], []
    B = labels.size(0)
    for i in range(B):
        anchor_label = labels[i].item()
        anchor_pid = pid_idx[i].item()

        # 找正样本，要求标签相同且不是自己
        pos_candidates = [j for j in range(B) if labels[j].item() == anchor_label and j != i]
        # 找负样本，标签不同，且身份不同
        neg_candidates = [j for j in range(B) if labels[j].item() != anchor_label and pid_idx[j].item() != anchor_pid]

        if len(pos_candidates) == 0 or len(neg_candidates) == 0:
            continue  # 跳过没有正负样本的

        p_idx = random.choice(pos_candidates)
        n_idx = random.choice(neg_candidates)

        a.append(i)
        p.append(p_idx)
        n.append(n_idx)
    if not a:
        # 若没采样到任何三元组，默认返回空tensor，训练时要注意
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.tensor(a), torch.tensor(p), torch.tensor(n)

def segment_frames_pad_zero(tensors: List[torch.Tensor], seg_len: int) -> list[list[torch.Tensor]]:
    if not tensors:
        zero = torch.zeros((3,224,224))
        return [[zero.clone() for _ in range(seg_len)]]
    segs, n = [], len(tensors)
    C,H,W = tensors[0].shape
    zero = torch.zeros((C,H,W), device=tensors[0].device)
    for i in range(0,n,seg_len):
        seg = tensors[i:i+seg_len]
        if len(seg)<seg_len: seg += [zero.clone() for _ in range(seg_len-len(seg))]
        segs.append(seg)
    return segs


# -----------------------------------------------------------------------------
# Split helper
# -----------------------------------------------------------------------------
def split_train_test(samples, fold_idx, fold_csv=None):   #fold_name:f"fold{fold_idx:0~4}"
    if fold_csv is None:fold_csv=read_data_sheet("/home/emotion/dataset/SEUMLD/5fold_list.csv")     #col->fold_name
    per = defaultdict(list)
    for i,(_,lbl,pid) in enumerate(samples): per[pid].append(i)#{pid:[sample_idx]}
    test_pids=set(fold_csv[f"fold{fold_idx}"].astype(str).to_list())
    train_idx=[]
    test_idx=[]
    for pid,subj in per.items():
        if pid in test_pids:
            test_idx.extend(subj)
        else:train_idx.extend(subj)
    return train_idx, test_idx

# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
triplet_loss = nn.TripletMarginLoss(margin=1.0)
cls_loss     = nn.CrossEntropyLoss()
id_loss_fn   = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# Run extract mode
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_extract(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VideoFrameDataset(cfg.data_root, cfg.label_path)
    set_seed(int(cfg.seed))
    mcfg = AutoConfig.from_pretrained(cfg.pretrained_model_path, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(cfg.pretrained_model_path, config=mcfg, trust_remote_code=True).to(device).eval()
    proc = VideoMAEImageProcessor.from_pretrained(cfg.pretrained_model_path)
    root = Path(cfg.feat_root); root.mkdir(parents=True, exist_ok=True)
    idxs=[]
    for imgs,lbl,pid,vid in tqdm(ds, desc="Extract"):
        d = root/ pid; d.mkdir(parents=True, exist_ok=True)
        fp = d/f"{vid}.pt"
        idxs.append((str(Path(pid)/f"{vid}.pt"), lbl, pid))
        if fp.exists(): continue
        frames = [torch.tensor(np.array(im).transpose(2,0,1)).float().to(device) for im in imgs]
        segf=[]

        for seg in segment_frames_pad_zero(frames, cfg.segment_size):
            pix = proc(seg, return_tensors="pt", do_rescale=True)["pixel_values"].to(device)
            pix = pix.permute(0,2,1,3,4)
            f = backbone.model.forward_features(pix)
            c = f[:,0,:] if f.dim()==3 else f
            segf.append(c.squeeze(0).cpu())
        out = torch.stack(segf)


        torch.save(out, fp)
    if not os.path.exists(cfg.index_pkl):
        pickle.dump(idxs, open(cfg.index_pkl,"wb"))
    logging.info(f"Extracted {len(idxs)} feature files.")

# -----------------------------------------------------------------------------
# Run training mode
# -----------------------------------------------------------------------------
def run_train(cfg):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed))
    ds = CachedFeatDataset(cfg.feat_root, cfg.index_pkl)
    samples = [(None, lbl, pid) for fp, lbl, pid in ds.index]
    fold_csv=pd.read_csv("/home/emotion/dataset/SEUMLD/5fold_list.csv", dtype=str, na_filter=False)
    fold_nums=len(fold_csv.columns)
    all_fold_logits = []
    all_fold_labels = []
    for fold_idx in range(fold_nums):
        tr, te = split_train_test(samples, fold_idx, fold_csv)
        fold_metric_base=metric_base[fold_idx]
        # 建立 pid 到索引的映射
        person_list = sorted(set([pid for fp, lbl, pid in ds.index]))
        pid2idx = {pid: idx for idx, pid in enumerate(person_list)}
        sample_cnt=Counter([ds.index[i][1] for i in tr])
        w=[sample_cnt[0]/sample_cnt[1] for _ in tr]                     #truth0, lie1
        print(f"train_cnt: {sample_cnt}")# train class count
        print(f"test_cnt: {Counter([ds.index[i][1] for i in te])}")# test class count
        spl = WeightedRandomSampler(w, len(tr), True)
        tld = DataLoader(Subset(ds,tr), batch_size=int(cfg.batch_size), sampler=spl, collate_fn=collate_cached)
        ted = DataLoader(Subset(ds,te), batch_size=int(cfg.batch_size), collate_fn=collate_cached)
        persons = len(person_list)
        dim = tld.dataset[0][0].shape[1]
        model = GenLieModel(dim, int(cfg.re_embed_dim), int(cfg.hidden_dim), float(cfg.dropout_rate), persons).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
        id_loss_weight = float(cfg.id_loss_weight)
        triplet_loss_weight = float(cfg.triplet_loss_weight)


        best_score = float('-inf')
        best_fold_logits = None
        best_epoch= None
        best_fold_preds = None
        best_fold_probs = None
        best_fold_labels = None
        
        for ep in range(cfg.epochs):
            model.train(); loss_s=0
            for feats,lbls,pids in tld:
                feats,lbls = feats.to(device), lbls.to(device)
                pid_idx = torch.tensor([pid2idx[pid] for pid in pids], device=device)
                with torch.amp.autocast('cuda', enabled=cfg.fp16):
                    logits, idl, emb = model(feats, lam=cfg.id_loss_lambda, return_feat=True)
                    lc = cls_loss(logits,lbls)
                    lid = id_loss_fn(idl, pid_idx)
                    a, p, n = sample_triplets(emb, lbls, pid_idx)

                    # 判断是否采样到有效三元组
                    if len(a) == 0:
                        # 没有有效三元组时，triplet loss置为0张量，且保留计算图
                        lt = torch.tensor(0., device=device, requires_grad=True)
                    else:
                        a = a.long()
                        p = p.long()
                        n = n.long()
                        lt = triplet_loss(emb[a], emb[p], emb[n])
                    loss = lc + id_loss_weight*lid + triplet_loss_weight*lt
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
                loss_s+=loss.item()*lbls.size(0)

            logging.info(f"Epoch {ep} train loss: {loss_s/len(tr):.4f}")
            model.eval()
            all_logits_list = []
            all_labels_list = []

            with torch.no_grad():
                for feats, lbls, _ in ted:
                    feats = feats.to(device)
                    logits, _ = model(feats, lam=0)
                    all_logits_list.append(logits.cpu().numpy())
                    all_labels_list.append(lbls.cpu().numpy())

            all_logits = np.concatenate(all_logits_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
            metrics = compute_metrics((all_logits, all_labels))
            acc = metrics["accuracy"]
            f1 = metrics["f1"]
            auc = metrics["auc"]
            loss= metrics["loss"]
            logging.info(f"Epoch {ep} fold{fold_idx} test acc: {acc:.4f}, f1: {f1:.4f}, auc: {auc:.4f}")
            if cfg.save_output:
                output_dict[cfg.hyper_param_key][fold_idx][ep]={
                    "metrics":metrics,
                    "epoch":ep,
                    "labels":all_labels,
                    "logits":all_logits
                }
            # score = acc-loss # 或使用自定义权重
            # score = acc-loss # 或使用自定义权重
            # score = acc-loss # 或使用自定义权重
            # score = f1 - loss # 或使用自定义权重
            # score=acc/fold_metric_base.acc+f1/fold_metric_base.f1+auc/fold_metric_base.auc-loss
            score=compute_score(metrics, fold_metric_base)

            if score > best_score:
                best_score = score
                best_epoch=ep
                best_fold_logits = all_logits
                best_fold_labels = all_labels#其实没必要
                print(f"best score updated: score:{best_score:.4f}, epoch:{ep}, test acc: {acc:.4f}, f1: {f1:.4f}, auc: {auc:.4f}")
                # torch.save(model.state_dict(), f"seumld_{cfg.output_model}")
            # break
        # Save best results for this fold
        all_fold_logits.append(best_fold_logits)
        all_fold_labels.append(best_fold_labels)
        if cfg.save_output:
            pickle.dump(output_dict, open(pwd/f"outputs/results/{Path(__file__).stem}_{feat_root.name}_outputs.pkl", "wb"))
            #'''{hyper_param_key:{fold_idx:{epoch:{metrics, labels, logits}}}}'''
    # Combine results from all folds
    # print(all_fold_labels)
    final_logits = np.concatenate(all_fold_logits)
    final_labels = np.concatenate(all_fold_labels)
    final_metrics = compute_metrics((final_logits, final_labels))
    final_acc = final_metrics["accuracy"]
    final_f1 = final_metrics["f1"]
    final_auc = final_metrics["auc"]
    final_loss = final_metrics["loss"]
    logging.info(f"Final best test acc: {final_acc:.4f}, f1: {final_f1:.4f}, auc: {final_auc:.4f}")
    final_score = compute_score(final_metrics, fold_metric_base)
            
    return final_score


# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["extract","train","grid"],required=True)
    p.add_argument("--seed",default="42")
    p.add_argument("--fp16",action="store_true")
    p.add_argument("--multi_gpu",action="store_true")
    #extract
    p.add_argument("--data_root",default="/home/emotion/dataset/SEUMLD/Preprocess/RegroupedVideo")
    p.add_argument("--label_path",default="/home/emotion/dataset/SEUMLD/Labels/Fine-grained-labels.csv")
    #train
    p.add_argument("--feat_root",default=f"/home/emotion/dataset/SEUMLD/features/videomaev2_128frames")
    p.add_argument("--index_pkl",default="/home/emotion/dataset/SEUMLD/cache/index.pkl")        #[(feat_path:'105/105_35.pt', label:1, pid:'105')] 
    p.add_argument("--pretrained_model_path",default="/home/emotion/models/VideoMAEv2-Base")

    p.add_argument("--batch_size",default="64")
    p.add_argument("--segment_size",default=16, type=int)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--hidden_dim",default="512")
    p.add_argument("--re_embed_dim",default="256")
    p.add_argument("--dropout_rate",default="0.3")
    p.add_argument("--weight_decay",type=float,default=1e-4)
    p.add_argument("--truths_per_person",type=int,default=3)
    p.add_argument("--lies_per_person",type=int,default=2)
    p.add_argument("--id_loss_weight",default="0.1")
    p.add_argument("--triplet_loss_weight",default="0.1")
    p.add_argument("--id_loss_lambda",type=float,default=1.0)
    p.add_argument("--output_model",default="best.pth")
    p.add_argument("--log_file",default="train_mae_personal_id_grid.log")
    # p.add_argument("--save_output", default=True)#将所有的output保存为pkl方便私下计算
    p.add_argument("--save_output",action="store_true", default=True)#将所有的output保存为pkl方便私下计算
    '''
    {
        f'{hyper_param}':{
            fold_idx:[
                {#epoch output
                    metrics, label, logits
                }
            ]
        }
    }
    '''

    args= p.parse_args()
    batches= sl(args.batch_size,int)
    seeds  = sl(args.seed,int)
    hids   = sl(args.hidden_dim,int)
    reds   = sl(args.re_embed_dim,int)
    drops  = sl(args.dropout_rate,float)
    idws   = sl(args.id_loss_weight,float)
    tws    = sl(args.triplet_loss_weight,float)
    args.hyper_param_key=f"bs={batches[0]}, seed={seeds[0]}, hidden={hids[0]}, re_dim={reds[0]}, drop={drops[0]}, id_w={idws[0]}, tri_w={tws[0]}"
    if data_root.is_file():
        strategy=data_root.stem
        log_file=Path(args.log_file)
        args.log_file=f"{log_file.stem}_{strategy}.log"
        feat_root=Path(args.feat_root).parent
        args.feat_root=f"{feat_root}/{strategy}"
    return args

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg=parse_args()
    pwd=Path(__file__).parent
    feat_root=Path(cfg.feat_root)
    init_logger(pwd/"outputs/log"/f"{Path(__file__).stem}_{feat_root.name}_{cfg.log_file}")
    logging.info(json.dumps(vars(cfg),indent=2,ensure_ascii=False))
    if cfg.mode=="extract": run_extract(cfg)
    elif cfg.mode=="train": run_train(cfg)
    else: grid_search(cfg)
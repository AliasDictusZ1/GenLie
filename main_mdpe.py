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
from utils.tools import choose_gpu, get_device
from utils.utils import init_logger, set_seed
os.environ["CUDA_VISIBLE_DEVICES"] = choose_gpu()
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pathlib import Path
from models.gen_lie import *
from utils.eval import *

device=get_device()

# -----------------------------------------------------------------------------
# Dataset: feature extraction
# -----------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    """Return (List[PIL.Image], label, pid, vid)"""
    def __init__(self, root_dir: str, labels_dir: str):
        self.root = Path(root_dir)
        self.labels = Path(labels_dir)
        assert self.root.is_dir() and self.labels.is_dir()
        self.samples: list[Tuple[Path,int,str,str]] = []
        self._build_index()

    def _build_index(self):
        for person in sorted(self.root.iterdir()):
            if not (person.is_dir() and re.fullmatch(r"\d{3}", person.name)): continue
            pid = person.name
            csv_f = self.labels / f"{pid}.csv"
            if not csv_f.exists(): continue
            df = pd.read_csv(csv_f); df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                q, qid = str(row["Question order"]).strip(), str(row["Question id"]).strip()
                vdir = person / f"{pid}-{q}-{qid}"
                if not vdir.exists(): continue
                lbl = 0 if int(row["Self evaluation of interviewee"]) == 0 else 1
                vid = f"{pid}-{q}-{qid}"
                self.samples.append((vdir, lbl, pid, vid))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vdir, lbl, pid, vid = self.samples[idx]
        imgs = [Image.open(p).convert("RGB") for p in sorted(vdir.glob("*.jpg"))]
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
        fp, lbl= self.index[idx]
        pid = Path(fp).parts[0] 
        feat = torch.load(self.root / fp)  # (T,D)
        return feat, torch.tensor(lbl), pid

def collate_cached(batch):
    feats, lbls, pids = zip(*batch)
    feats = torch.stack([f.mean(0) for f in feats])  # (B,D)
    return feats, torch.stack(lbls), list(pids)


# -----------------------------------------------------------------------------
# Split helper
# -----------------------------------------------------------------------------
def split_train_test(samples, truths_per=3, lies_per=2, seed=42):
    rng = np.random.default_rng(seed)
    per = defaultdict(list)
    for i,(_,lbl,pid) in enumerate(samples): per[pid].append(i)
    test=[]
    for pid, idxs in per.items():
        t=[i for i in idxs if samples[i][1]==0]  # 注意：truth标签现在是0
        l=[i for i in idxs if samples[i][1]==1]  # 注意：lie标签现在是1
        rng.shuffle(t); rng.shuffle(l)
        test += t[:truths_per] + l[:lies_per]
    train = list(set(range(len(samples))) - set(test))
    return train, test

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
    ds = VideoFrameDataset(cfg.data_root, cfg.labels_dir)
    set_seed(int(cfg.seed))
    mcfg = AutoConfig.from_pretrained(cfg.pretrained_model_path, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(cfg.pretrained_model_path, config=mcfg, trust_remote_code=True).to(device).eval()
    proc = VideoMAEImageProcessor.from_pretrained(cfg.pretrained_model_path)
    root = Path(cfg.feat_root); root.makedirs(exist_ok=True)
    idxs=[]
    for imgs,lbl,pid,vid in tqdm(ds, desc="Extract"):
        frames = [torch.tensor(np.array(im).transpose(2,0,1)).float().to(device) for im in imgs]
        segf=[]
        for seg in segment_frames_pad_zero(frames, cfg.segment_size):
            pix = proc(seg, return_tensors="pt", do_rescale=True)["pixel_values"].to(device)
            pix = pix.permute(0,2,1,3,4)
            f = backbone.model.forward_features(pix)
            c = f[:,0,:] if f.dim()==3 else f
            segf.append(c.squeeze(0).cpu())
        out = torch.stack(segf)
        d = root/ pid; d.makedirs(exist_ok=True)
        fp = d/f"{vid}.pt"
        torch.save(out, fp)
        idxs.append((str(Path(pid)/f"{vid}.pt"), lbl, pid))
    pickle.dump(idxs, open(cfg.index_pkl,"wb"))
    logging.info(f"Extracted {len(idxs)} feature files.")

# -----------------------------------------------------------------------------
# Run training mode
# -----------------------------------------------------------------------------


def run_train(cfg):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed))
    ds = CachedFeatDataset(cfg.feat_root, cfg.index_pkl)
    samples = [(None, lbl, Path(fp).parts[0]) for fp, lbl in ds.index]
    tr, te = split_train_test(samples, cfg.truths_per_person, cfg.lies_per_person, int(cfg.seed))
    # 建立 pid 到索引的映射
    person_list = sorted(set([Path(fp).parts[0] for fp, lbl in ds.index]))
    pid2idx = {pid: idx for idx, pid in enumerate(person_list)}
    sample_cnt=Counter([ds.index[i][1] for i in tr])
    w=[sample_cnt[1]/sample_cnt[0] for _ in tr]
    print(f"train_cnt: {sample_cnt}")
    print(f"test_cnt: {Counter([ds.index[i][1] for i in te])}")
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


    best_score = 0.0
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
        model.eval(); all_logits = []; all_labels = []
        with torch.no_grad():
            for feats, lbls, _ in ted:
                feats = feats.to(device)
                logits, _ = model(feats, lam=0)
                all_logits.append(logits.cpu())
                all_labels.append(lbls)
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        metrics = compute_metrics((all_logits, all_labels))
        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        auc = metrics["auc"]
        logging.info(f"Epoch {ep} test acc: {acc:.4f}, f1: {f1:.4f}, auc: {auc:.4f}")

        score = 0.5 * acc + 0.25 * auc + 0.25 * f1 # 或使用自定义权重

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), cfg.output_model)
    return best_score

# -----------------------------------------------------------------------------
# Grid search mode
# -----------------------------------------------------------------------------
def grid_search(cfg):
    def sl(v, t): return [t(x) for x in v.split(',')]
    batches= sl(cfg.batch_size,int)
    seeds  = sl(cfg.seed,int)
    hids   = sl(cfg.hidden_dim,int)
    reds   = sl(cfg.re_embed_dim,int)
    drops  = sl(cfg.dropout_rate,float)
    idws   = sl(cfg.id_loss_weight,float)
    tws    = sl(cfg.triplet_loss_weight,float)
    combos= list(product(batches,seeds,hids,reds,drops,idws,tws))
    results=[]; best=0; bc=None
    for bs,sd,h,rd,dr,idw,tw in combos:
        logging.info(f"网格搜索参数: bs={bs}, seed={sd}, hidden={h}, re_dim={rd}, drop={dr}, id_w={idw}, tri_w={tw}")
        cfg.batch_size=str(bs); cfg.seed=str(sd)
        cfg.hidden_dim=str(h); cfg.re_embed_dim=str(rd)
        cfg.dropout_rate=str(dr)
        cfg.id_loss_weight=str(idw); cfg.triplet_loss_weight=str(tw)
        acc = run_train(cfg)
        results.append((acc,bs,sd,h,rd,dr,idw,tw))
        if acc>best: best,bc=acc,(bs,sd,h,rd,dr,idw,tw)
    out=Path(cfg.feat_root)/"grid_results.csv"
    with open(out,'w') as f:
        w=csv.writer(f); w.writerow(["acc","bs","seed","hid","re","drop","idw","tw"])
        w.writerows(results)
    logging.info(f"Best: {best:.4f} {bc}")
    return best

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["extract","train","grid"],required=True)
    p.add_argument("--seed",default="42")
    p.add_argument("--fp16",action="store_true")
    p.add_argument("--multi_gpu",action="store_true")

    p.add_argument("--data_root",default="/home/emotion/dataset/mdpe_128_faces")
    p.add_argument("--labels_dir",default="/home/emotion/dataset/MDPE/labels")
    p.add_argument("--feat_root",default="/home/emotion/dataset/MDPE/features/Video_MAEV2/data")
    p.add_argument("--index_pkl",default="/home/emotion/dataset/MDPE/features/Video_MAEV2/data/index_t0l1.pkl")
    p.add_argument("--pretrained_model_path",default="/home/emotion/models/VideoMAEv2-Base")

    p.add_argument("--batch_size",default="64")
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
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg=parse_args()
    pwd=Path(__file__).parent
    feat_root=Path(cfg.feat_root)
    init_logger(pwd/"outputs/log"/f"{feat_root.name}_{Path(__file__).stem}_{cfg.log_file}")
    logging.info(json.dumps(vars(cfg),indent=2,ensure_ascii=False))
    if cfg.mode=="extract": run_extract(cfg)
    elif cfg.mode=="train": run_train(cfg)
    else: grid_search(cfg) 
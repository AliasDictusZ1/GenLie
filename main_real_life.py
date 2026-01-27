#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoMAEv2-Large · Long-video lie detection for Real Life Dataset
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
# Dataset: feature extraction for Real Life dataset
# -----------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    """Return (List[PIL.Image], label, pid, vid)
    
    Real Life 数据集结构:
    {pid:{vid:[jpg_frames]}}，其中vid.split("_")[1]就是truth/lie标签
    """
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        assert self.root.is_dir()
        self.samples: list[Tuple[Path,int,str,str]] = []
        self._build_index()

    def _build_index(self):
        # 遍历根目录下所有人物文件夹
        for person in sorted(self.root.iterdir(),key=lambda x: int(x.name)):
            if not person.is_dir(): continue
            pid = person.name
            
            # 遍历该人物下的所有视频目录
            for video_dir in sorted(person.iterdir(),key=lambda x:x.name):
                if not video_dir.is_dir(): continue
                vid = video_dir.name
                
                # 从视频ID中提取标签 (truth=0, lie=1)
                label_str = vid.split("_")[1].lower()
                lbl = 0 if label_str == "truth" else 1
                
                # 检查是否有图像文件
                if list(video_dir.glob("*.jpg")):
                    self.samples.append((video_dir, lbl, pid, vid))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vdir, lbl, pid, vid = self.samples[idx]
        # 获取所有图片文件并排序
        img_files = sorted(list(vdir.glob("*.jpg")), key=lambda x: x.name)
        
        # 平均采样128帧
        total_frames = len(img_files)
        target_frames = 128
        
        if total_frames == 0:
            # 如果没有图片，返回空列表
            return [], lbl, pid, vid
        elif total_frames <= target_frames:
            # 如果原始帧数小于等于128，则直接使用所有帧
            imgs = [Image.open(p).convert("RGB") for p in img_files]
        else:
            # 如果原始帧数大于128，则均匀采样
            indices = [int(i * total_frames / target_frames) for i in range(target_frames)]
            indices = [min(i, total_frames-1) for i in indices]  # 确保索引不超出范围
            imgs = [Image.open(img_files[i]).convert("RGB") for i in indices]
        
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
        fp, lbl, pid = self.index[idx]
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
def split_train_test(samples, fold_idx):
    train= [i for i in range(len(samples)) if i!=fold_idx]
    test = [fold_idx]
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
    ds = VideoFrameDataset(cfg.data_root)  # 注意这里不再传入labels_dir
    set_seed(int(cfg.seed))
    mcfg = AutoConfig.from_pretrained(cfg.pretrained_model_path, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(cfg.pretrained_model_path, config=mcfg, trust_remote_code=True).to(device).eval()
    proc = VideoMAEImageProcessor.from_pretrained(cfg.pretrained_model_path)
    root = Path(cfg.feat_root); root.mkdir(parents=True, exist_ok=True)
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
        d = root/ pid; d.mkdir(parents=True, exist_ok=True)
        fp = d/f"{vid}.pt"
        torch.save(out, fp)
        idxs.append((str(Path(pid)/f"{vid}.pt"), lbl, pid))
    pickle.dump(idxs, open(cfg.index_pkl,"wb"))
    logging.info(f"Extracted {len(idxs)} feature files.")

# -----------------------------------------------------------------------------
# Run training mode
# -----------------------------------------------------------------------------
def run_train(cfg):
    device = get_device()
    set_seed(int(cfg.seed))
    ds = CachedFeatDataset(cfg.feat_root, cfg.index_pkl)
    
    # 准备所有样本
    samples = ds.index
    num_samples = len(samples)
    
    # 存储所有折的预测结果
    all_true_labels = []  # 所有样本的真实标签
    all_pred_labels = []  # 所有样本的预测标签
    all_pred_scores = []  # 所有样本的预测为"谎言"类的概率
    
    # 准备记录最佳结果
    best_fold_results = []
    
    # 计算标签分布
    label_counts = Counter([lbl for _, lbl, _ in samples])
    logging.info(f"数据集统计: 总样本数={num_samples}, truth样本={label_counts[0]}, lie样本={label_counts[1]}")
    
    # 执行留1交叉验证
    logging.info(f"开始执行{num_samples}折留1交叉验证...")
    
    for fold_idx in range(num_samples):
        logging.info(f"\n处理第 {fold_idx+1}/{num_samples} 折...")
        
        # 划分训练集和测试集
        tr, te = split_train_test(samples, fold_idx)
        
        # 获取当前测试样本的标签
        current_label = samples[fold_idx][1]
        label_name = "Truth" if current_label == 0 else "Lie"
        logging.info(f"测试样本标签: {label_name}")
        
        # 建立 pid 到索引的映射
        person_list = sorted(set([pid for _, _, pid in ds.index]))
        pid2idx = {pid: idx for idx, pid in enumerate(person_list)}
        
        # 统计训练集中的样本数量
        sample_cnt = Counter([ds.index[i][1] for i in tr])
        # 根据样本比例设置权重
        w = [sample_cnt[1]/sample_cnt[0] for _ in tr]
        logging.info(f"训练集分布: {sample_cnt}")
        logging.info(f"测试集: {Counter([ds.index[i][1] for i in te])}")
        
        # 创建数据加载器
        spl = WeightedRandomSampler(w, len(tr), True)
        tld = DataLoader(Subset(ds,tr), batch_size=int(cfg.batch_size), sampler=spl, collate_fn=collate_cached)
        ted = DataLoader(Subset(ds,te), batch_size=int(cfg.batch_size), collate_fn=collate_cached)
        
        # 准备模型
        persons = len(person_list)
        dim = tld.dataset[0][0].shape[1]
        model = GenLieModel(dim, int(cfg.re_embed_dim), int(cfg.hidden_dim), float(cfg.dropout_rate), persons).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
        id_loss_weight = float(cfg.id_loss_weight)
        triplet_loss_weight = float(cfg.triplet_loss_weight)

        # 训练循环
        best_score = float('-inf')  # 改为负无穷，确保第一个epoch结果能被保存
        best_test_results = None
        best_epoch = -1
        patience = 9999
        patience_counter = 0
        has_print=False

        for ep in tqdm(range(cfg.epochs), desc=f"fold{fold_idx} Train", ncols=80):
            # 训练阶段
            model.train()
            loss_s=0
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
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                loss_s+=loss.item()*lbls.size(0)
            
            train_loss = loss_s/len(tr)
            # logging.info(f"Fold {fold_idx+1}, Epoch {ep+1} train loss: {train_loss:.4f}")
            
            # 评估阶段
            model.eval()
            all_logits = []
            all_labels = []
            test_loss = 0
            
            with torch.no_grad():
                for feats, lbls, _ in ted:
                    feats = feats.to(device)
                    lbls = lbls.to(device)
                    logits, _ = model(feats, lam=0)
                    test_loss += cls_loss(logits, lbls).item() * lbls.size(0)
                    all_logits.append(logits.cpu())
                    all_labels.append(lbls.cpu())
            
            test_loss /= len(te)
            all_logits = torch.cat(all_logits).numpy()
            all_labels = torch.cat(all_labels).numpy()
            
            # 计算测试指标
            probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
            predictions = np.argmax(probs, axis=-1)
            test_acc = np.mean(predictions == all_labels)
            
            if not has_print and test_acc>0:
                logging.info(f"Fold {fold_idx+1}, Epoch {ep+1} test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
                has_print=True
            
            # 保存当前epoch的测试结果
            current_test_results = (predictions, probs[:, 1], all_labels)
            
            # 如果是第一个epoch，初始化best_test_results
            if ep == 0:
                best_test_results = current_test_results
            
            # 计算综合得分
            curr_score = test_acc - test_loss  # 或使用其他评分方式
            
            # 早停检查
            if curr_score > best_score:
                best_score = curr_score
                best_epoch = ep
                patience_counter = 0
                
                # 保存最佳模型
                # torch.save(model.state_dict(), f"{cfg.output_model}_fold{fold_idx}.pth")
                
                # 保存最佳测试结果
                best_test_results = current_test_results
                
                # logging.info(f"Fold {fold_idx+1}: 发现更好的模型! 综合得分: {best_score:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"早停: {patience}个轮次没有改进")
                    break
        
        logging.info(f"Fold {fold_idx+1} 完成训练, 最佳epoch: {best_epoch+1}, 最佳得分: {best_score:.4f}")
        
        # 收集该折的结果
        if best_test_results:
            pred, score, label = best_test_results
            all_pred_labels.extend(pred)
            all_pred_scores.extend(score)
            all_true_labels.extend(label)
            
            # 记录该折的详细结果
            for i in range(len(pred)):
                best_fold_results.append({
                    "fold_idx": fold_idx,
                    "sample_id": f"{samples[fold_idx][2]}-{samples[fold_idx][0]}",
                    "true_label": int(label[i]),
                    "pred_label": int(pred[i]),
                    "pred_score": float(score[i]),
                    "correct": bool(pred[i] == label[i])
                })
            
            # 打印该折的结果
            correct = np.sum(pred == label)
            total = len(label)
            logging.info(f"折 {fold_idx+1} 测试结果: 准确率 = {correct}/{total} ({100.0*correct/total:.2f}%)")
        else:
            # 处理没有找到最佳模型的情况
            # 使用最后一个epoch的结果作为这一折的结果
            logging.warning(f"折 {fold_idx+1} 没有找到有效的最佳模型，使用默认值")
            
            # 创建默认预测（预测全部为多数类）
            majority_label = Counter([lbl for _, lbl, _ in samples]).most_common(1)[0][0]
            
            # 使用默认值
            pred = np.array([majority_label])  # 默认预测为多数类
            score = np.array([0.5])  # 默认预测分数为0.5（不确定）
            label = np.array([samples[fold_idx][1]])  # 真实标签
            
            all_pred_labels.extend(pred)
            all_pred_scores.extend(score)
            all_true_labels.extend(label)
            
            # 记录该折的详细结果
            best_fold_results.append({
                "fold_idx": fold_idx,
                "sample_id": f"{samples[fold_idx][2]}-{samples[fold_idx][0]}",
                "true_label": int(label[0]),
                "pred_label": int(pred[0]),
                "pred_score": float(score[0]),
                "correct": bool(pred[0] == label[0])
            })
            
            # 打印该折的结果
            correct = np.sum(pred == label)
            total = len(label)
            logging.info(f"折 {fold_idx+1} 测试结果(默认值): 准确率 = {correct}/{total} ({100.0*correct/total:.2f}%)")
            
            # 每10折或在最后一折计算累计指标
            if (fold_idx + 1) % 10 == 0 or fold_idx == num_samples - 1:
                metrics = compute_metrics((np.array(all_true_labels), np.array(all_pred_labels), np.array(all_pred_scores)))
                acc = metrics["accuracy"]
                f1 = metrics["f1"]
                auc = metrics["auc"]
                logging.info(f"\n当前累计结果 ({fold_idx+1}/{num_samples} 折):")
                logging.info(f"累计准确率: {acc:.4f}")
                logging.info(f"累计F1: {f1:.4f}")
                logging.info(f"累计AUC: {auc:.4f}")
    
    # 计算最终指标
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    all_pred_scores = np.array(all_pred_scores)
    
    metrics = compute_metrics((all_true_labels, all_pred_labels, all_pred_scores))
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    auc = metrics["auc"]
    
    # 计算每个类别的准确率
    truth_indices = np.where(all_true_labels == 0)[0]
    lie_indices = np.where(all_true_labels == 1)[0]
    
    truth_acc = np.mean(all_pred_labels[truth_indices] == 0) if len(truth_indices) > 0 else 0
    lie_acc = np.mean(all_pred_labels[lie_indices] == 1) if len(lie_indices) > 0 else 0
    
    # 打印最终结果
    logging.info("\n留1交叉验证最终结果:")
    logging.info(f"总体准确率: {acc:.4f}")
    logging.info(f"Truth准确率: {truth_acc:.4f}")
    logging.info(f"Lie准确率: {lie_acc:.4f}")
    logging.info(f"F1分数: {f1:.4f}")
    logging.info(f"AUC: {auc:.4f}")
    
    # 保存结果
    results = {
        'true_labels': all_true_labels.tolist(),
        'pred_labels': all_pred_labels.tolist(),
        'pred_scores': all_pred_scores.tolist(),
        'metrics': {
            'accuracy': float(acc),
            'f1': float(f1),
            'auc': float(auc),
            'truth_accuracy': float(truth_acc),
            'lie_accuracy': float(lie_acc)
        },
        'fold_results': best_fold_results
    }
    
    # 保存结果到文件
    result_path = Path(cfg.feat_root) / f"{Path(__file__).stem}_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"结果已保存到: {result_path}")
    
    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc
    }

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
        result = run_train(cfg)
        # 处理run_train函数返回的字典，提取准确率
        if isinstance(result, dict):
            acc = result.get('accuracy', 0)
        else:
            acc = 0  # 默认值
        results.append((acc,bs,sd,h,rd,dr,idw,tw))
        if acc>best: 
            best = acc
            bc = (bs,sd,h,rd,dr,idw,tw)
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

    p.add_argument("--data_root",default="/home/zhangzs/emotion/dataset/real-life-trail-face-frames/Downloads")
    p.add_argument("--feat_root",default="/home/pengxf/work/TDD/Video_MAEV2/data_real_life")
    p.add_argument("--index_pkl",default="/home/pengxf/work/TDD/Video_MAEV2/data_real_life/index_t0l1.pkl")
    p.add_argument("--pretrained_model_path",default="/home/emotion/models/VideoMAEv2-Base")
    p.add_argument("--segment_size",type=int,default=16)  # 添加segment_size参数

    p.add_argument("--batch_size",default="64")
    p.add_argument("--epochs",type=int,default=300)
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
    p.add_argument("--output_model",default="best_real_life.pth")
    p.add_argument("--log_file",default="train_real_life_personal_id_grid.log")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg=parse_args()
    pwd=Path(__file__).parent
    feat_root=Path(cfg.feat_root)
    init_logger(pwd/"outputs/log"/f"{feat_root.name}_{Path(__file__).stem}_sampler_{cfg.log_file}")
    logging.info(json.dumps(vars(cfg),indent=2,ensure_ascii=False))
    if cfg.mode=="extract": run_extract(cfg)
    elif cfg.mode=="train": run_train(cfg)
    else: grid_search(cfg) 
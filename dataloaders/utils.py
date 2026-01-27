#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载工具函数
"""
from collections import defaultdict, Counter
import random
from typing import List
import numpy as np
import torch
from pathlib import Path

def segment_frames_pad_zero(tensors: List[torch.Tensor], seg_len: int) -> list[list[torch.Tensor]]:
    """
    将视频帧分段，不足的部分用零张量填充
    """
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

def split_train_test(samples, truths_per=3, lies_per=2, seed=42):
    """
    按照每人固定数量的真实/谎言视频划分训练集和测试集
    """
    rng = np.random.default_rng(seed)
    per = defaultdict(list)
    for i,(_,lbl,pid) in enumerate(samples): per[pid].append(i)
    test=[]
    for pid, idxs in per.items():
        t=[i for i in idxs if samples[i][1]==1]
        l=[i for i in idxs if samples[i][1]==0]
        rng.shuffle(t); rng.shuffle(l)
        test += t[:truths_per] + l[:lies_per]
    train = list(set(range(len(samples))) - set(test))
    return train, test 

def sl(v, t): return [t(x) for x in v.split(',')]
# -----------------------------------------------------------------------------
# Grid search mode
# -----------------------------------------------------------------------------
def grid_search(cfg):

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
        cfg.hyper_param_key=f"bs={bs}, seed={sd}, hidden={h}, re_dim={rd}, drop={dr}, id_w={idw}, tri_w={tw}"
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
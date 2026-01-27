#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存特征数据集加载模块
"""
import pickle
from pathlib import Path
import torch
from torch.utils.data import Dataset

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
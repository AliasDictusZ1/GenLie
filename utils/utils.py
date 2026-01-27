#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数模块
"""
import random
import logging
import numpy as np
import torch
from pathlib import Path

def init_logger(log_file: str= None):
    """
    初始化日志记录器
    """
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handlers = [logging.StreamHandler()]
    log_file=Path(log_file)
    log_file.mkdir(parents=True, exist_ok=True)
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

def set_seed(seed: int):
    """
    设置随机种子，确保实验可重复
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
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
        output_dict[cfg.hyper_param_key]=defaultdict(dict)#{fold_idx:{spoch:{}}}
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
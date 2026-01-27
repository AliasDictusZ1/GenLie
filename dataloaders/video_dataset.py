#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频帧数据集加载模块
"""
from pathlib import Path
import re
from typing import Tuple
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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
                lbl = 1 if int(row["Self evaluation of interviewee"]) == 0 else 0
                vid = f"{pid}-{q}-{qid}"
                self.samples.append((vdir, lbl, pid, vid))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vdir, lbl, pid, vid = self.samples[idx]
        imgs = [Image.open(p).convert("RGB") for p in sorted(vdir.glob("*.jpg"))]
        return imgs, lbl, pid, vid 
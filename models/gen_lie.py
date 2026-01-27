import os, json, argparse, random, logging, pickle, re
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Gradient Reversal Layer
# -----------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None

def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)

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
# Model: cached features training
# -----------------------------------------------------------------------------
class ReEmbedEncoder(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=256, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden=512, drop=0.3, num_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, num_cls)
        )
    def forward(self, x): return self.net(x)

class GenLieModel(nn.Module):
    def __init__(self, feat_dim, re_dim, hidden, drop, num_persons):
        super().__init__()
        self.re = ReEmbedEncoder(feat_dim, hidden, re_dim, drop)
        self.cls = ClassifierHead(re_dim, hidden, drop, num_cls=2)
        self.idh = ClassifierHead(re_dim, hidden, drop, num_cls=num_persons)

    def forward(self, feats, lam=1.0, return_feat=False):
        emb = self.re(feats)
        logits = self.cls(emb)
        idlog = self.idh(grad_reverse(emb, lam))
        return (logits, idlog, emb) if return_feat else (logits, idlog)



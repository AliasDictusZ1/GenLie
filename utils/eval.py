import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

output_dict=defaultdict(lambda:defaultdict(dict)) #用于保存seumld所有的输出，方便自己私下计算

def compute_metrics(logits, labels):
    """
    计算评估指标：准确率、F1和AUC
    
    Args:
        eval_pred: 可以是(logits, labels)二元组或(true_labels, pred_labels, pred_scores)三元组
    
    Returns:
        包含各种指标的字典
    """
    # print('compute_metrics',logits.shape,labels.shape)
    loss=F.cross_entropy(torch.tensor(logits), torch.tensor(labels))
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=-1)
    lie_scores = probs[:, 1]  # 预测为lie类的概率

    # 计算准确率
    acc = accuracy_score(labels, predictions)
    
    # 计算 F1 分数
    f1 = f1_score(labels, predictions, average='binary',zero_division=0)  # 若为二分类    
    
    # 计算 AUC
    if len(np.unique(labels)) == 2 and len(np.unique(lie_scores)) == 2:
        auc = roc_auc_score(labels, lie_scores)
    else:
        # 如果AUC计算失败(比如只有一个类别)，返回0.5
        auc = 0.5
        
    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "loss": loss.item()
    }

#seumld base score limit
metric_base={0: {'acc': 0.6562, 'f1': 0.523, 'auc': 0.6435}, 1: {'acc': 0.6571, 'f1': 0.5102, 'auc': 0.5917}, 2: {'acc': 0.6687, 'f1': 0.513, 'auc': 0.581}, 3: {'acc': 0.6682, 'f1': 0.3947, 'auc': 0.5506}, 4: {'acc': 0.6577, 'f1': 0.504, 'auc': 0.5231}}

def compute_score(metric, fold_metric_base):
    score=metric["accuracy"]/fold_metric_base["acc"]+metric["f1"]/fold_metric_base["f1"]+metric["auc"]/fold_metric_base["auc"]-metric["loss"]
    return score
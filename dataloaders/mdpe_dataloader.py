import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import numpy as np
import pickle
from tools import *
import random
from collections import Counter,defaultdict

pwd=get_current_root(__file__)
label_dict={"truth":0, "lie":1}
subj_ids=[f"{vid:03d}" for vid in range(1,194)]
frame_dir=Path(f"/home/emotion/dataset/mdpe_full_face")
cache_dir=Path(f"/home/emotion/dataset/MDPE/data_cache")

def padding(samples):
    max_len=max([len(sample) for sample in samples])
    padded_samples=[]
    padding_masks=[]
    # samples=[torch.cat([])]) for sample in samples]

    for sample in samples:
        padded_samples.append(
            torch.cat([torch.tensor(sample), torch.zeros(max_len-len(sample),sample.shape[-1])])
            )
        padding_masks.append(torch.concat((torch.ones(len(sample)),torch.zeros(max_len-len(sample)))))
    return torch.stack(padded_samples),torch.stack(padding_masks)      #torch.stack(samples)

def collate_fn(samples):
    keys=next(iter(samples)).keys()
    batch={key:[sample[key] for sample in samples] for key in keys}

    batch["label"]=torch.tensor([label>0 for label in batch["label"]]).long()
    # batch["label"]=torch.tensor([label==0 for label in batch["label"]]).long()          #颠倒定义，f1处于0.75左右
    batch["visual"]= torch.stack([torch.tensor(sample).float().mean(dim=0) for sample in batch["visual"]]) #padding(batch["visual"])
    # batch["visual"],batch["padding_mask"]=padding(batch["visual"])
    # return batch
    return batch["visual"],batch["label"],batch["vid"]

def load_annotations():
    label_dir=Path("/home/emotion/dataset/MDPE/labels")
    transcript_dir=Path("/home/emotion/dataset/MDPE/transcriptions")
    annotation={vid:{} for vid in subj_ids}
    for vid in subj_ids:
        label_path=label_dir/f"{vid}.csv"
        labels=read_data_sheet(label_path)# col[0]-qorder col[1]-qid
        for idx, sample_label in labels.iterrows():
            sample_id=f"{vid}-{sample_label['Question order']}-{sample_label['Question id']}"
            transcript_path=transcript_dir/f"{vid}/{sample_id}.txt"
            annotation[vid][sample_id]={
                "vid":vid,
                "sample_id":sample_id,
                "label": 0 if sample_label["Self evaluation of interviewee"]==0 else 1,
                "text": read_text_list(transcript_path)[0]
            }
    return annotation

def load_feat_dir(feat_dir):
    feat_dict=defaultdict(dict)
    for subj_id in subj_ids:
        subj_feat_dir=feat_dir/f"{subj_id}"
        for feat_file in subj_feat_dir.iterdir():
            if not feat_file.is_file():continue
            # print(feat_file.stem.split(".")[0])
            feat_dict[subj_id][feat_file.stem.split(".")[0]]=torch.load(feat_file) if feat_file.suffix=="pt" else np.load(feat_file)
    return feat_dict

def uniform_sample_frames(frames, max_frames=128):
    """
    对帧进行均匀采样，最多获取max_frames个样本。
    
    参数:
        frames (list): 包含所有帧路径的列表。
        max_frames (int): 需要采样的最大帧数。
    
    返回:
        list: 均匀采样后的帧路径列表。
    """
    total_frames = len(frames)
    
    # 如果帧数不足或正好等于max_frames，则直接返回所有帧
    if total_frames <= max_frames:
        return frames
    # 否则进行均匀采样
    indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    sampled_frames = [frames[i] for i in indices]
    
    return sampled_frames

def load_frames(frame_dir=frame_dir, num_frames=128):
    frame_dict=defaultdict(dict)
    for subj_id in subj_ids:
        subj_frame_dir=frame_dir/f"{subj_id}"
        # print(list(subj_frame_dir.iterdir()))
        for sample_path in subj_frame_dir.iterdir():
            sample_id=sample_path.stem
            if sample_path.is_file():continue
            frame_paths=uniform_sample_frames(sorted(sample_path.glob('*.jpg'),key=lambda x:x.stem),num_frames)
            # print(frame_paths)
            frame_dict[subj_id][sample_id]=frame_paths
    return frame_dict

def load_data(feature_path):
    feature_path=Path(feature_path)
    feat_dict=pickle.load(open(feature_path, "rb")) if feature_path.is_file() else load_feat_dir(feature_path)
    annotation=load_annotations()
    train_data, test_data=[], []
    for subj_id,subj in annotation.items():
        lie_cnt=2
        truth_cnt=3
        sample_list=[(sample_id,sample) for sample_id,sample in subj.items()]
        random.shuffle(sample_list)
        for sample_id,sample in sample_list:
            sample=annotation[subj_id][sample_id]
            sample["visual"]=feat_dict[subj_id][sample_id]
            label=sample["label"]

            if label==0 and truth_cnt>0:    #test truth
                truth_cnt-=1
                test_data.append(sample)
            elif label>0 and lie_cnt>0:     # test lie
                lie_cnt-=1
                test_data.append(sample)
            else:                           #train
                train_data.append(sample)

    return train_data, test_data

class MDPEDataset(Dataset):
    def __init__(self, data):
        self.data=data
        self.len=len(self.data)
        self.stats_dict=None
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.len
    
    def stats(self):
        if self.stats_dict is None:
            self.stats_dict={}
            for sample in self.data:
                label_key=f"{'truth'if sample['label']==0 else 'lie'}"
                if label_key not in self.stats_dict:
                    self.stats_dict[label_key]=0
                self.stats_dict[label_key]+=1
        return self.stats_dict

def get_dataloader(batch_size=4, feature_path=""):
    train_data, test_data=load_data(feature_path)
    train_dataset=MDPEDataset(train_data)
    test_dataset=MDPEDataset(test_data)
    
    # 获取训练集统计信息
    stats = train_dataset.stats()
    print(f"训练集统计: {stats}")
    
    # 直接使用main_mdpev3.py中的采样器实现
    labels = [sample["label"] for sample in train_data]
    
    # 计算类别权重 - 正确实现（暂时注释掉）
    # class_counts = Counter(labels)
    # class_weights = {0: 1.0/class_counts[0], 1: 1.0/class_counts[1]}
    # 
    # # 为每个样本分配对应类别的权重
    # w = [class_weights[label] for label in labels]
    
    # 原始main_mdpev3.py中的实现
    w = [Counter(labels)[0]/Counter(labels)[1] for _ in range(len(train_data))]
    
    # 创建加权采样器
    sampler = None #WeightedRandomSampler(w, len(w), True)
    
    # 使用采样器创建训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # print(f"启用了加权采样，权重比例: {Counter(labels)[1]/Counter(labels)[0]}")
    
    return train_loader, test_loader


if __name__=="__main__":
    train_loader,test_loader=get_dataloader(feature_path="/home/emotion/dataset/MDPE/data_cache/videomae_b4_me_top15.pkl")
    for item in train_loader:
        print(item)
        pass
        # print(item["visual"].shape)

    for item in test_loader:
        print(item)
        # print(item["visual"].shape)
        pass
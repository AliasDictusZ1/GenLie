from pathlib import Path
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from collections import defaultdict

pwd=Path(__file__).parent
cache_dir=Path("/home/emotion/dataset/real_life/Real-life_Deception_Detection_2016/cache")
cue_path=cache_dir/"facial_cues.pkl"
feat_path=cache_dir/"vit_fps1.pkl"
frame_root=Path(f"/home/emotion/dataset/real-life-trail-face-frames/Downloads")

label_list=['truth','lie']
label_dict={
    label:idx
    for idx,label in enumerate(label_list)
}
label_dict[0]=label_dict['truth']
label_dict[1]=label_dict['lie']

def load_feat():
    return pickle.load(open(feat_path, "rb"))

def load_annotations():
    annotation=pickle.load(open(cue_path, "rb"))
    for subj_id,subj in annotation.items():
        for sample_id,sample_cue in subj.items():
            annotation[subj_id][sample_id]={
                "label":label_dict[sample_id.split('_')[1]],  #truth/lie
                "cues":sample_cue                       
            }
    return annotation

def load_data():
    annotation=load_annotations()
    feat=load_feat()

    for subj_id,subj in feat.items():
        for sample_id,sample_feat in subj.items():
            # print(annotation[subj_id][sample_id]["label"])
            feat[subj_id][sample_id]={
                "label":annotation[subj_id][sample_id]["label"],
                "visual":sample_feat["visual"],
                "cues":annotation[subj_id][sample_id]["cues"]
            }

    # print(next(iter(next(iter(feat.values())).values())).keys())
    # print(next(iter(next(iter(annotation.values())).values())).keys())
    return feat
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

def load_frames(frame_dir=frame_root, num_frames=128):
    frame_dict=defaultdict(dict)
    for subj_dir in frame_dir.iterdir():
        subj_id=subj_dir.stem
        subj_frame_dir=frame_dir/f"{subj_id}"
        if subj_frame_dir.is_file():continue
        for sample_path in subj_frame_dir.iterdir():
            sample_id=sample_path.stem
            if sample_path.is_file():continue
            frame_paths=uniform_sample_frames(sorted(sample_path.iterdir(),key=lambda x:x.stem),num_frames)
            # print(frame_paths)
            frame_dict[subj_id][sample_id]=frame_paths
    return frame_dict

class RealLifeDataset(Dataset):
    def __init__(self, data: List[Dict], transform=None):
        """
        Initialize MDPE Dataset
        
        Args:
            data: List of data items, each containing visual features, cues, and label
            transform: Optional transform to be applied on the data
        """
        self.data = data
        self.transform = transform
        self.len = len(data)
        
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        # print(item,item['visual'])
        # Convert numpy arrays to torch tensors
        feature = torch.from_numpy(item['visual']).float()
        label = item['label']
        AUs = torch.from_numpy(item['cues']).float()  # Shape: (59,)
        
        if self.transform:
            feature = self.transform(feature)
            AUs = self.transform(AUs)
            
        return {
            'feature': feature,
            'label': label,
            'AUs': AUs
        }
def padding_batch(sample_list, pad_value=0.0):
    """
    Pad a batch of tensors with shape (T, *) to (B, T_max, *)
    Automatically handles variable-length sequences with multi-dim tokens.

    Args:
        sample_list: list of tensors with shape (T, *)
        pad_value: value to pad with

    Returns:
        padded_tensor: (B, T_max, *)
        lengths: (B,)
    """
    lengths = [sample.shape[0] for sample in sample_list]
    max_len = max(lengths)
    sample_shape = sample_list[0].shape[1:]  # keep tail dims

    padded_shape = (len(sample_list), max_len, *sample_shape)
    padded_tensor = torch.full(padded_shape, pad_value)

    for i, sample in enumerate(sample_list):
        T = sample.shape[0]
        padded_tensor[i, :T] = sample  # broadcasting works with any tail dims

    return padded_tensor, torch.tensor(lengths, dtype=torch.long)

def collate_fn(batch):
    keys=batch[0].keys()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch={key:[sample[key] for sample in batch] for key in keys}

    batch["feature"]=[sample[:, :1, :]if len(sample.shape)==3 else sample for sample in batch["feature"]]#use cls token
    batch["feature"], batch["feature_seqlen"] = padding_batch(batch["feature"])
    # print(batch["feature"].shape)
    # print(batch["feature"].shape)
    # sampling_rate=batch["sample_rate"][0]

    #也许在外部提取以启用num_workers
    # batch["audio"],batch["audio_cls"]=audio_encoder(batch["audio"])
    # batch["text"],batch["text_cls"]=text_encoder(batch["text"])
    batch['label'] = torch.tensor(batch['label']).long()
    batch['AUs'] = torch.stack(batch['AUs'])
    return batch

def get_dataloaders(batch_size,fold_idx, data=load_data(), transform=None):
    data_list=[sample for subj_id,subj in data.items() for sample_id,sample in subj.items()] 
    train_loader=DataLoader(RealLifeDataset([data for idx,data in enumerate(data_list) if idx!=fold_idx],transform), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader=DataLoader(RealLifeDataset([data_list[fold_idx]],transform), batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

if __name__=="__main__":
    print({subj_id:[sample_id for sample_id in subj.keys()] for subj_id,subj in load_frames().items()})
    exit()

    # print(load_data())
    train_loader, test_loader=get_dataloaders(64,0)
    for batch in train_loader:
        print(batch)
    for batch in test_loader:
        print(batch)
    
                
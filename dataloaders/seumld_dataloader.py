from pathlib import Path
from tools import *
import pandas as pd
from collections import defaultdict
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm

cache_dir=Path("/home/emotion/dataset/SEUMLD/cache")
label_df_path=Path("/home/emotion/dataset/SEUMLD/Labels/Fine-grained-labels.csv")
fold_df_path=Path("/home/emotion/dataset/SEUMLD/5fold_list.csv")
dataset_dir=Path(f"/home/emotion/dataset/SEUMLD")

class SeumldDataset(Dataset):
    def __init__(self,data_list):
        self.data=data_list
        self.len=len(data_list)
    def __getitem__(self,idx):
        item=self.data[idx]
        # item["visual"]=item["visual"].mean(0)
        return item
    def __len__(self):
        return self.len

def load_feat(feat_dir:str):
    if isinstance(feat_dir,str):feat_dir=Path(feat_dir)
    cache_path=cache_dir/f"{feat_dir.stem}.pkl"
    if cache_path.exists():return pickle.load(open(cache_path,"rb"))
    data_dict=defaultdict(dict)
    for subj_dir in feat_dir.iterdir():
        subj_id=subj_dir.stem
        for feat_path in subj_dir.iterdir():
            sample_id=feat_path.stem
            feat=torch.load(feat_path)
            data_dict[subj_id][sample_id]=feat
    pickle.dump(data_dict,open(cache_path,"wb"))
    return data_dict

def load_annotations():
    label_df=pd.read_csv(label_df_path, dtype=str, na_filter=False)
    annotations=defaultdict(dict)
    for _,sample in label_df.iterrows():            
        sample_id=sample['name']               #sample_id
        subj_id=sample_id.split('_')[0]
        label=sample['label']                  #sample.label/sample['label']: 0:truth/1:lie
        annotations[subj_id][sample_id]={
            "label":int(label),
            'subj_id':subj_id
        }
    return annotations

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

def load_frames(use_raw=False, num_frames=128):
    frame_dir=dataset_dir/(f"Preprocess/RegroupedVideo" if use_raw else f"frames_clip")
    frame_dict=defaultdict(dict)
    for subj_dir in frame_dir.iterdir():
        subj_id=subj_dir.stem
        subj_frame_dir=frame_dir/f"{subj_id}"
        if subj_frame_dir.is_file() or not subj_frame_dir.exists():continue
        for sample_path in subj_frame_dir.iterdir():
            sample_id=sample_path.stem
            if sample_path.is_file():continue
            frame_paths=uniform_sample_frames(sorted(sample_path.iterdir(),key=lambda x:x.stem),num_frames)
            # print(frame_paths)
            frame_dict[subj_id][sample_id]=frame_paths
    return frame_dict

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
    return torch.stack(padded_samples),torch.stack(padding_masks).bool()      #torch.stack(samples)

def collate_fn(batch):
    keys=batch[0].keys()#dict_keys(['label', 'subj_id', 'visual'])
    batch={key:[sample[key] for sample in batch] for key in keys}
    feats, lbls, pids = batch['visual'], batch['label'], batch['subj_id']
    feats = torch.stack([torch.tensor(f).mean(0) for f in feats])  # (B,D)
    # feats,feats_mask=padding(feats)
    # return feats,feats_mask, torch.stack(lbls), list(pids)
    return feats, torch.tensor(lbls), list(pids)

def get_fold_df():
    return pd.read_csv(fold_df_path, dtype=str, na_filter=False)

def split_annotation_list(fold_idx, annotations=load_annotations(), fold_df=get_fold_df()):
    #get fold subj_id
    test_subjs=set(fold_df[f"fold{fold_idx}"].astype(str).to_list())
    test_data=[]
    train_data=[]
    # merge data+label then split
    annotations=load_annotations()
    for subj_id,subj in annotations.items():
        (test_data if subj_id in test_subjs else train_data).extend([sample for sample in subj.values()])
    return train_data, test_data

def split_annotation_dict(fold_idx, annotations=load_annotations(), fold_df=get_fold_df()):
    #get fold subj_id
    test_subjs=set(fold_df[f"fold{fold_idx}"].astype(str).to_list())
    test_data={}
    train_data={}
    # merge data+label then split
    annotations=load_annotations()
    for subj_id,subj in annotations.items():
        (test_data if subj_id in test_subjs else train_data)[subj_id]=subj
    return train_data, test_data

def get_dataloaders(batch_size, fold_idx, data_dict=load_feat("/home/emotion/dataset/SEUMLD/features/videomaev2_128frames"),fold_df=pd.read_csv(fold_df_path, dtype=str, na_filter=False)):
    '''
    data_dict={subj_id:{sample_id:feature.shape([seq_len,feat_dim])}}
    annotations={subj_id:{sample_id:{label:t0/l1}}}
    '''
    #get fold subj_id
    test_subjs=set(fold_df[f"fold{fold_idx}"].astype(str).to_list())
    test_data=[]
    train_data=[]
    
    # merge data+label then split
    annotations=load_annotations()
    for subj_id,subj in data_dict.items():
        for sample_id,sample_feat in subj.items():
            annotations[subj_id][sample_id]["visual"]=sample_feat['visual']#vit
            # annotations[subj_id][sample_id]["visual"]=sample_feat
        (test_data if subj_id in test_subjs else train_data).extend([sample for sample in annotations[subj_id].values()])
    # print(train_data,test_data)
    #get dataloaderr
    train_loader=DataLoader(SeumldDataset(train_data),batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader=DataLoader(SeumldDataset(test_data),batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return train_loader,test_loader

if __name__ =="__main__":
    fold_df=get_fold_df()
    fold_num=len(fold_df.columns)
    data_dict=load_feat("/home/emotion/dataset/SEUMLD/features/videomaev2_raw_128frames")
    for fold_idx in range(fold_num):
        train_loader,test_loader=get_dataloaders(batch_size=64, data_dict=data_dict, fold_idx=fold_idx, fold_df=fold_df)
        for batch in tqdm(train_loader, desc=f"Train fold{fold_idx}", ncols=80):
            print(batch.visual.shape)
        
        for batch in tqdm(test_loader, desc=f"Eval fold{fold_idx}", ncols=80):
            print(batch.visual.shape)
         

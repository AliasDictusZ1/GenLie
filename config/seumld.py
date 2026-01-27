from config import *
from pathlib import Path

dataset_dir=Path("/home/emotion/dataset/SEUMLD")
feature_root=dataset_dir/"features"
frame_root=dataset_dir/"Preprocess/RegroupedVideo"
label_path = dataset_dir/"Labels/Fine-grained-labels.csv"
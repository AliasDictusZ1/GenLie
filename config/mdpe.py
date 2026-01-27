from config import *
from pathlib import Path

dataset_dir=Path("/home/emotion/dataset/MDPE")
feature_root=dataset_dir/"features"
frame_root=dataset_dir.parent/"mdpe_128_faces"
label_dir=dataset_dir/"labels"
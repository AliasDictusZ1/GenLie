from pathlib import Path

proj_root=Path(__file__).parent.parent      # GenLie
pretrained_model_path=Path("/home/emotion/models/VideoMAEv2-Base")

output_root= proj_root/"output"
output_root.mkdir(parents=True, exist_ok=True)

num_classes=2
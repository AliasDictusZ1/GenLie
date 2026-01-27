# GenLie: A Global-Enhanced Lie Detection Network under Sparsity and Semantic Interference

## Overview

GenLie is a deep learning framework for video-based deception detection. It leverages the VideoMAEv2 model to extract features from video frames and employs a combination of classification, identity-adversarial, and triplet loss to detect deceptive behavior in videos. The framework supports multiple datasets, including MDPE, SEUMLD, and Real Life Trialdatasets.

## Features

- Video feature extraction using VideoMAEv2-Base model
- Multi-modal deception detection
- Identity-adversarial training to reduce person-specific biases
- Triplet loss for better feature representation
- Support for multiple datasets (MDPE, SEUMLD, Real Life Trial)
- Comprehensive evaluation metrics (accuracy, F1 score, AUC)

## Project Structure

```
├── config/                 # Configuration files for different datasets
│   ├── common.py           # Common configuration settings
│   ├── mdpe.py             # MDPE dataset configuration
│   ├── real_life.py        # Real Life dataset configuration
│   └── seumld.py           # SEUMLD dataset configuration
├── dataloaders/            # Data loading modules
│   ├── cached_dataset.py   # Dataset for cached features
│   ├── mdpe_dataloader.py  # MDPE dataset loader
│   ├── real_life_dataloader.py # Real Life dataset loader
│   ├── seumld_dataloader.py # SEUMLD dataset loader
│   ├── tools.py            # Data loading utilities
│   └── video_dataset.py    # Video frame dataset loader
├── main_mdpe.py            # Main script for MDPE dataset
├── main_real_life.py       # Main script for Real Life dataset
├── main_seumld.py          # Main script for SEUMLD dataset
├── models/                 # Model definitions
│   └── gen_lie.py          # GenLie model implementation
├── utils/                  # Utility functions
│   ├── eval.py             # Evaluation metrics
│   ├── logger.py           # Logging utilities
│   ├── tools.py            # General utilities
│   └── utils.py            # Miscellaneous utilities
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd GenLie
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the pretrained VideoMAEv2 model and update the path in `config/common.py`.

## Usage

The framework supports three main modes of operation:

### 1. Feature Extraction

Extract and cache CLS features from video frames:

```bash
python main_mdpe.py --mode extract
# or
python main_seumld.py --mode extract
# or
python main_real_life.py --mode extract
```

### 2. Model Training

Train the classifier with identity-adversarial and triplet loss on cached features:

```bash
python main_mdpe.py --mode train
# or
python main_seumld.py --mode train
# or
python main_real_life.py --mode train
```

## Model Architecture

The GenLie model consists of three main components:

1. **Feature Re-embedding**: Transforms the extracted video features into a more discriminative embedding space.
2. **Classification Head**: Predicts whether the input video contains deceptive behavior (lie=1) or truthful behavior (truth=0).
3. **Identity-Adversarial Head**: Reduces person-specific biases by adversarially training against person identity.

The model is trained with a combination of:
- Classification loss (Cross-Entropy)
- Identity-adversarial loss
- Triplet loss for better feature representation

## Datasets

The framework supports three datasets:

1. **MDPE**: A deception detection dataset with structured interviews.
2. **SEUMLD**: Southeast University Multimodal Lie Detection dataset.
3. **Real Life**: A dataset with real-life deception scenarios.

## Evaluation

The model's performance is evaluated using multiple metrics:
- Accuracy
- F1 Score
- Area Under the ROC Curve (AUC)

## License

[Specify the license here]

## Citation

[Provide citation information if applicable]

## Acknowledgements

- VideoMAEv2 for video feature extraction
- [Add other acknowledgements as needed]
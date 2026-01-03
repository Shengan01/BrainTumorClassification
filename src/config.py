import torch
import os

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths - Organized structure
BASE_EXPERIMENTS_DIR = "experiments"
EXPERIMENTS_DIR = os.path.join(BASE_EXPERIMENTS_DIR, "models")
CV_DIR = os.path.join(BASE_EXPERIMENTS_DIR, "cv_results")
STATS_DIR = os.path.join(BASE_EXPERIMENTS_DIR, "stats")
VISUALIZATIONS_DIR = "visualizations"

# Data
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_CLASSES = 4
SEED = 42
N_FOLDS = 3
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def set_seed(seed=SEED):
    """Set all random seeds for full reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Training - Custom Models (From Scratch)
EPOCHS_CUSTOM = 150
LR_CUSTOM = 3e-4
PATIENCE_CUSTOM = 15  # More patience for training from scratch

# Training - Pre-trained Models (Fine-tuning)
EPOCHS_PRETRAINED = 20
LR_PRETRAINED = 1e-4
PATIENCE_PRETRAINED = 5  # Less patience - should converge fast

# Shared
MIN_DELTA = 0.001
WEIGHT_DECAY = 1e-4

# Model Specifics
HIDDEN_DIM = 256

# Visualization Thresholds (higher = more focused/less noise)
CNN_VIZ_PERCENTILE = 90   # Keep top 15% of CNN activations
GRAD_VIZ_PERCENTILE = 90  # Keep top 15% of gradient/saliency maps

# Ensure directories exist
for d in [EXPERIMENTS_DIR, CV_DIR, STATS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(d, exist_ok=True)

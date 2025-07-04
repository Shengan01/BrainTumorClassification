# BrainTumorClassification

This project implements a deep learning pipeline for brain tumor classification using a hybrid architecture that combines Convolutional Neural Networks (CNNs) with Transformer Encoders. It also compares performance against ViT and ResNet-based baselines.

📁 Dataset

    Source: Kaggle - Brain Tumor MRI Dataset

    Classes: Glioma, Meningioma, Pituitary, No Tumor

    Split: Training, Validation (10%), and Testing

Dataset is automatically downloaded using kagglehub:
```
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```

🧠 Model Architectures
🔷 Hybrid Model (Custom)

    CNN Tokenizer with CBAM (Channel + Spatial Attention)

    Transformer Encoder (multi-head attention + positional embedding)

    Attention Pooling Head

    Final linear classifier

🔶 ViT & ResNet Baselines

    Images converted to 3-channel

    Loaded using torchvision/timm pretrained models

    Compared against custom hybrid model


🏋️‍♂️ Training Details

    Batch Size: 64

    Epochs: 200

    Learning Rate: 1e-4

    Optim: Adam

    Scheduler: Cosine Annealing

    Loss: CrossEntropy with label smoothing

    Mixed Precision: Enabled via torch.amp

    Early Stopping: Patience = 25 epochs
    
🎛️ Data Augmentation

Different pipelines are used for:

    Hybrid model: grayscale 1-channel input

    ViT/ResNet models: grayscale expanded to 3-channel

Augmentations include:

    Random crops and flips

    Rotation and affine transformations

    Gaussian noise and blur

    Color jitter

    Random erasing

📊 Evaluation

    Metrics: Accuracy, Loss

    Best model saved based on validation loss

    Test performance reported with accuracy

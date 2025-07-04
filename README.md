BrainTumorClassification

This project implements a deep learning pipeline for brain tumor classification using a hybrid architecture that combines Convolutional Neural Networks (CNNs) with Transformer Encoders. It also compares performance against several state-of-the-art models like ViT (Vision Transformer), ResNet, DenseNet, RegNet, EfficientNet, ConvNeXt, Swin Transformer, and MaxViT.

📁 Dataset

    Source: Kaggle - Brain Tumor MRI Dataset

    Classes: Glioma, Meningioma, Pituitary, No Tumor

    Split: Training, Validation (10%), and Testing

The dataset is automatically downloaded using kagglehub:
```
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```

🧠 Model Architectures
🔷 Hybrid Model (Custom)

    CNN Tokenizer with CBAM (Channel + Spatial Attention)

    Transformer Encoder: Multi-head attention + positional embedding

    Attention Pooling Head

    Final Linear Classifier

🔶 Vision Transformer (ViT) & ResNet Baselines

    Images: Converted to 3-channel format

    Models: Pretrained on ImageNet, using torchvision/timm for ViT, and ResNet-based architectures.

    Compared against: Custom Hybrid model, as well as other models below.

🔶 Other Models (Pretrained)

    DenseNet121

    RegNetY-032

    EfficientNetV2

    ConvNeXt

    Swin Transformer

    MaxViT

All models are evaluated against the same dataset for a fair comparison.
🏋️‍♂️ Training Details

    Batch Size: 64

    Epochs: 200

    Learning Rate: 1e-4

    Optimizer: AdamW (for all models)

    Scheduler: Cosine Annealing with T_max=200

    Loss Function: CrossEntropy Loss with label smoothing (0.01)

    Mixed Precision: Enabled via torch.amp for faster training

    Early Stopping: Patience = 15 epochs (triggered based on validation loss)

🎛️ Data Augmentation

    Hybrid model: Grayscale 1-channel input

    ViT/ResNet models: Grayscale expanded to 3-channel

    Augmentations:

        Random crops and flips

        Rotation and affine transformations

        Gaussian noise and blur

        Color jitter

        Random erasing

📊 Evaluation
Metrics:

    Accuracy: Primary metric for model evaluation.

    Additional Metrics:

        Precision (Macro Avg.)

        Recall (Macro Avg.)

        F1-Score (Macro Avg.)

        AUC (Area Under the Curve)

        Confusion Matrix: To better understand misclassifications

Model Selection:

    The best model is selected based on the lowest validation loss during training.

    Test performance is reported with accuracy and additional metrics (precision, recall, F1-score, AUC).

Test Results:

    All models are evaluated using a common test set.

    Results include detailed performance metrics for each model.

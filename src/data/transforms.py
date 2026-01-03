import torch
from torchvision import transforms

# ImageNet normalization for pre-trained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_transforms(img_size=224, channels=1):
    """
    Get transforms for training and testing.
    - channels=1: Grayscale with 0.5 mean/std (for custom Hybrid model)
    - channels=3: RGB with ImageNet stats (for pre-trained baselines)
    
    Both pipelines use IDENTICAL geometric augmentations.
    """
    
    # Base augmentations (Geometric only - safe for MRI)
    # Resize ensures we don't crop out pathology.
    augmentations = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
    ]

    if channels == 1:
        # Grayscale for Hybrid model
        mean, std = [0.5], [0.5]
        train_transforms_list = [
            transforms.Grayscale(num_output_channels=1),
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        test_transforms_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    else:
        # RGB with ImageNet normalization for pre-trained models
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        train_transforms_list = [
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        test_transforms_list = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    return transforms.Compose(train_transforms_list), transforms.Compose(test_transforms_list)

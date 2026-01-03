import os
import kagglehub
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, random_split
from src.data.transforms import get_transforms
from src.config import IMG_SIZE

def get_data_path():
    print("Checking/Downloading dataset path...")
    try:
        path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        print(f"Dataset path: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return 'brain-tumor-mri-dataset' # Fallback


def get_sartaj_data_path():
    """Download sartajbhuvaji brain tumor classification dataset (4 classes)."""
    print("Checking/Downloading Sartaj dataset...")
    try:
        path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
        print(f"Sartaj dataset path: {path}")
        return path
    except Exception as e:
        print(f"Error downloading Sartaj dataset: {e}")
        return None


def get_pkdarabi_data_path():
    """Download pkdarabi medical image dataset (4 classes)."""
    print("Checking/Downloading PKDarabi dataset...")
    try:
        path = kagglehub.dataset_download("pkdarabi/medical-image-dataset-brain-tumor-detection")
        print(f"PKDarabi dataset path: {path}")
        return path
    except Exception as e:
        print(f"Error downloading PKDarabi dataset: {e}")
        return None


def get_external_test_dataset(data_dir, img_size=None, channels=1):
    """
    Load an external dataset for evaluation (test only).
    Auto-detects structure: Training/Testing folders or direct class folders.
    """
    if img_size is None:
        img_size = IMG_SIZE
    _, test_transform = get_transforms(img_size, channels)
    
    # Try common folder structures
    test_dir = os.path.join(data_dir, 'Testing')
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_dir, 'testing')
    if not os.path.exists(test_dir):
        # Try 'Test' folder
        test_dir = os.path.join(data_dir, 'Test')
    if not os.path.exists(test_dir):
        # Maybe the data_dir itself contains class folders
        test_dir = data_dir
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Could not find test data in {data_dir}")
    
    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
    print(f"Loaded external dataset from {test_dir}: {len(test_ds)} images, {len(test_ds.classes)} classes")
    print(f"Classes: {test_ds.classes}")
    return test_ds

def get_datasets(data_dir, img_size=None, channels=1, seed=42):
    if img_size is None:
        img_size = IMG_SIZE  # Use config value
    train_transform, test_transform = get_transforms(img_size, channels)
    
    # Check if directories exist
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    
    # Load separate instances for train (aug) and validation (clean)
    full_dataset_train_aug = datasets.ImageFolder(train_dir, transform=train_transform)
    full_dataset_val_clean = datasets.ImageFolder(train_dir, transform=test_transform)
    
    # Split
    total_size = len(full_dataset_train_aug)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(total_size), [train_size, val_size], generator=generator)
    
    # Create Subsets
    train_ds = Subset(full_dataset_train_aug, train_indices.indices)
    val_ds = Subset(full_dataset_val_clean, val_indices.indices) # Uses clean dataset with val indices
    
    if os.path.exists(test_dir):
        test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
    else:
        print("Warning: Test directory not found. Using Val set as Test set.")
        test_ds = val_ds

    return train_ds, val_ds, test_ds

def get_dataloaders(train_ds, val_ds, test_ds, batch_size=64, num_workers=2, shuffle_test=False):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True 
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=shuffle_test, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

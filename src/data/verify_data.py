import torch
import torch.nn as nn
from src.data.dataset import get_datasets, get_dataloaders, get_data_path
import matplotlib.pyplot as plt
import os
import numpy as np

def verify_data():
    print("Verifying data loading...")
    data_dir = get_data_path()
    
    # 1. Verify Datasets
    print("\n--- Checking Datasets ---")
    train_ds, val_ds, test_ds = get_datasets(data_dir, img_size=224, channels=1) # Check 1 channel mainly
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")
    
    # Check indices disjointness
    train_indices = set(train_ds.indices)
    val_indices = set(val_ds.indices)
    intersection = train_indices.intersection(val_indices)
    print(f"Intersection between Train and Val indices: {len(intersection)} (Should be 0)")
    assert len(intersection) == 0, "Train and Val indices overlap!"
    
    # 2. Verify Dataloaders
    print("\n--- Checking Dataloaders ---")
    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=16)
    
    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape} (Should be [16, 1, 224, 224])")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    
    # Check normalization
    print(f"Min pixel value: {images.min():.4f}")
    print(f"Max pixel value: {images.max():.4f}")
    print(f"Mean pixel value: {images.mean():.4f}")
    
    # 3. Visualize Augmentations
    print("\n--- Saving Visualization ---")
    # Denormalize for visualization (approximate since we used 0.5 mean/std)
    # img = (img * std) + mean
    vis_images = images * 0.5 + 0.5 
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = vis_images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {labels[i].item()}")
        
    plt.suptitle("Augmented Training Samples")
    plt.tight_layout()
    plt.savefig("verification_samples.png")
    print("Saved 'verification_samples.png'")
    
    print("\nSUCCESS: Data verification complete.")

if __name__ == "__main__":
    verify_data()

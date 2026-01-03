import matplotlib.pyplot as plt
import numpy as np
import os
from src.config import VISUALIZATIONS_DIR

def imshow_samples(loader, title="Sample Images"):
    """
    Shows a batch of images from the loader.
    """
    images, labels = next(iter(loader))
    images = images[:16] # Show up to 16
    
    plt.figure(figsize=(12, 12))
    # Un-normalize
    # Assuming 0.5 mean/std as per config
    images = images * 0.5 + 0.5
    
    for i in range(len(images)):
        plt.subplot(4, 4, i+1)
        img = images[i].numpy().transpose((1, 2, 0))
        if img.shape[2] == 1:
            img = img.squeeze()
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"eda_{title.lower().replace(' ', '_')}.png"))
    plt.close()

def plot_class_distribution(train_ds, val_ds, test_ds, class_names):
    """
    Plots the count of samples per class for each split.
    """
    # Helper to get counts
    def get_counts(ds):
        # If subset, access underlying targets
        if hasattr(ds, 'dataset'):
            targets = [ds.dataset.targets[i] for i in ds.indices]
        else:
            targets = ds.targets
        
        counts = np.bincount(targets, minlength=len(class_names))
        return counts

    train_counts = get_counts(train_ds)
    val_counts = get_counts(val_ds)
    test_counts = get_counts(test_ds)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, train_counts, width, label='Train')
    plt.bar(x, val_counts, width, label='Val')
    plt.bar(x + width, test_counts, width, label='Test')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Dataset Distribution per Class')
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "dataset_distribution.png"))
    plt.close()

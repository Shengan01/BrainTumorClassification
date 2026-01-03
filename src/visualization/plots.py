import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import numpy as np
import os
from src.config import VISUALIZATIONS_DIR

def plot_metrics_per_model(models_stats, save_dir=None):
    """
    Plot training curves (loss, accuracy, etc.) for multiple models.
    models_stats: {'ModelName': {'train_acc': [...], 'val_acc': [...], ...}}
    """
    if save_dir is None:
        save_dir = VISUALIZATIONS_DIR
    
    metrics = ['train_acc', 'val_acc', 'train_loss', 'val_loss']
    titles = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric, title in zip(axes, metrics, titles):
        for model_name, stats in models_stats.items():
            if metric in stats:
                ax.plot(stats[metric], label=model_name, linewidth=2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves_all_models.png"), dpi=150)
    plt.close()

def plot_confusion_matrix(cm, class_names, model_name="Model", accuracy=None, auc=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # Build title with metrics
    title = f"{model_name}"
    if accuracy is not None and auc is not None:
        subtitle = f"Accuracy: {accuracy:.2%} | AUC: {auc:.4f}"
        plt.title(f"{title}\n{subtitle}", fontweight='bold', fontsize=12)
    else:
        plt.title(title, fontweight='bold', fontsize=12)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    safe_name = model_name.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"confusion_matrix_{safe_name}.png"), dpi=150)
    plt.close()

def display_predictions_with_probabilities(model, test_loader, device, class_names, num_per_class=3):
    """Display grid of predictions with probability bars - one row per class."""
    import torch
    model.eval()
    
    # Collect samples per class
    class_images = {i: [] for i in range(len(class_names))}
    
    for images, labels in test_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            if len(class_images[label]) < num_per_class:
                class_images[label].append((images[i], label))
        if all(len(v) >= num_per_class for v in class_images.values()):
            break
    
    # Flatten and organize by class (rows = classes)
    rows = len(class_names)
    cols = num_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    
    for row_idx, class_idx in enumerate(range(len(class_names))):
        samples = class_images[class_idx]
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                img_tensor, true_label = samples[col_idx]
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred = outputs.argmax(dim=1).item()
                
                img = img_tensor.squeeze(0).cpu().numpy()
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(np.transpose(img, (1, 2, 0)))
                
                true_name = class_names[true_label]
                pred_name = class_names[pred]
                conf = probs.max() * 100
                
                color = 'green' if pred == true_label else 'red'
                ax.set_title(f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)", 
                             color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "predictions_with_probabilities.png"), dpi=150)
    plt.close()

def create_architecture_diagram():
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    fig.suptitle('Hybrid CNN-Transformer Architecture', fontsize=16, fontweight='bold', y=0.98)
    
    colors = {
        'input': '#FFE5B4', 'cnn': '#87CEEB', 'attention': '#FFB6C1',
        'transformer': '#C8E6C9', 'head': '#F0E68C', 'output': '#E1BEE7'
    }
    
    def draw_box(x, y, w, h, title, content, color):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
                              edgecolor='black', facecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x, y + h/2 - 0.25, title, ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(x, y - 0.05, content, ha='center', va='center', fontsize=8, family='monospace')
    
    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=25,
                                linewidth=2.5, color='darkblue')
        ax.add_patch(arrow)
    
    y = 2.5
    positions = [1.2, 3.5, 5.8, 8.1, 10.4, 12.7, 15, 17.3]
    box_width = 1.6
    
    draw_box(positions[0], y, box_width, 1, 'INPUT', '(B, 1, 224, 224)', colors['input'])
    draw_box(positions[1], y, box_width, 1.1, 'CNN TOKENIZER', 'ResBlocks\n1→32→64→128', colors['cnn'])
    draw_arrow(positions[0] + box_width/2, y, positions[1] - box_width/2, y)
    draw_box(positions[2], y, box_width, 1, 'ATTENTION', 'Channel+Spatial', colors['attention'])
    draw_arrow(positions[1] + box_width/2, y, positions[2] - box_width/2, y)
    draw_box(positions[3], y, box_width, 1, 'TOKENIZATION', '(B, 196, 256)', colors['input'])
    draw_arrow(positions[2] + box_width/2, y, positions[3] - box_width/2, y)
    draw_box(positions[4], y, box_width, 1.1, 'TRANSFORMER', '4L, 4H', colors['transformer'])
    draw_arrow(positions[3] + box_width/2, y, positions[4] - box_width/2, y)
    draw_box(positions[5], y, box_width, 1.1, 'HEAD', 'AttnPool→LN→FC', colors['head'])
    draw_arrow(positions[4] + box_width/2, y, positions[5] - box_width/2, y)
    draw_box(positions[6], y, box_width, 0.9, 'LOGITS', '(B, 4)', colors['output'])
    draw_arrow(positions[5] + box_width/2, y, positions[6] - box_width/2, y)
    draw_box(positions[7], y, box_width, 0.9, 'OUTPUT', 'Softmax', colors['input'])
    draw_arrow(positions[6] + box_width/2, y, positions[7] - box_width/2, y)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "hybrid_architecture.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_final_results_table(results_list, save_name="final_results_table"):
    """
    Create a visual table of final test results.
    results_list: list of dicts with 'Model', 'Test_Accuracy', 'Test_AUC'
    """
    if not results_list:
        print("No results to plot.")
        return
    
    import pandas as pd
    df = pd.DataFrame(results_list)
    df = df.sort_values(by='Test_Accuracy', ascending=False).reset_index(drop=True)
    
    # Format for display
    df['Accuracy'] = df['Test_Accuracy'].apply(lambda x: f"{x:.2%}")
    df['AUC'] = df['Test_AUC'].apply(lambda x: f"{x:.4f}")
    display_df = df[['Model', 'Accuracy', 'AUC']]
    
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 1))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4CAF50'] * 3
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('lightgrey')
    
    plt.title("Final Test Results (Sorted by Accuracy)", fontweight='bold', fontsize=12, pad=20)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Results table saved to {VISUALIZATIONS_DIR}/{save_name}.png")

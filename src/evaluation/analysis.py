from thop import profile
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.config import VISUALIZATIONS_DIR, STATS_DIR

def analyze_model_complexity(model, input_res=(1, 224, 224), device='cuda'):
    """Analyze model complexity using thop. Returns (macs, params) or (None, None) on failure."""
    try:
        model = model.to(device).eval()
        # Create dummy input with correct shape
        dummy_input = torch.randn(1, *input_res).to(device)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return macs, params
    except Exception as e:
        print(f"Complexity analysis failed for {model.__class__.__name__}: {e}")
        return None, None

def compare_efficiency(models_dict, device='cuda'):
    """
    Compare multiple models efficiency with VERTICAL bar chart.
    models_dict: {'ModelName': (model_instance, input_channels)}
    """
    results = []
    print("\nComparing Efficiency across models...")
    
    for name, (model, in_channels) in models_dict.items():
        input_res = (in_channels, 224, 224)
        macs, params = analyze_model_complexity(model, input_res=input_res, device=device)
        if macs is not None:
            results.append({
                'Model': name,
                'GFLOPs': macs / 1e9,
                'Params_M': params / 1e6
            })
            print(f"  {name}: {macs/1e9:.2f} GFLOPs, {params/1e6:.1f}M params")
    
    if not results:
        print("No models analyzed successfully.")
        return None
        
    df = pd.DataFrame(results)
    # Round to 2 decimal places for cleaner output
    df['GFLOPs'] = df['GFLOPs'].round(2)
    df['Params_M'] = df['Params_M'].round(2)
    # Sort by GFLOPs for the first chart
    df_flops = df.sort_values(by='GFLOPs', ascending=True).reset_index(drop=True)
    # Sort by Params for the second chart
    df_params = df.sort_values(by='Params_M', ascending=True).reset_index(drop=True)
    df.to_csv(os.path.join(STATS_DIR, "efficiency_comparison.csv"), index=False)
    
    # Get Hybrid as baseline
    hybrid_row = df[df['Model'] == 'Hybrid']
    baseline_flops = hybrid_row['GFLOPs'].values[0] if not hybrid_row.empty else df['GFLOPs'].min()
    baseline_params = hybrid_row['Params_M'].values[0] if not hybrid_row.empty else df['Params_M'].min()
    
    # Create VERTICAL bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot 1: GFLOPs (sorted by GFLOPs)
    n = len(df_flops)
    x = np.arange(n)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))
    
    ax1 = axes[0]
    bars1 = ax1.bar(x, df_flops['GFLOPs'], color=colors, edgecolor='black', width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_flops['Model'], rotation=45, ha='right')
    ax1.set_ylabel('GFLOPs (lower is better)', fontsize=11)
    ax1.set_title('Computational Cost', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add labels on top of bars
    for i, (bar, val) in enumerate(zip(bars1, df_flops['GFLOPs'])):
        pct = ((val - baseline_flops) / baseline_flops) * 100 if baseline_flops > 0 else 0
        if df_flops['Model'].iloc[i] == 'Hybrid':
            label = f"{val:.2f}\n(baseline)"
        else:
            label = f"{val:.2f}\n({pct:+.0f}%)"
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                 label, ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Parameters (sorted by Params)
    colors2 = plt.cm.viridis(np.linspace(0.2, 0.8, n))
    ax2 = axes[1]
    bars2 = ax2.bar(x, df_params['Params_M'], color=colors2, edgecolor='black', width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_params['Model'], rotation=45, ha='right')
    ax2.set_ylabel('Parameters (Millions)', fontsize=11)
    ax2.set_title('Model Size', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars2, df_params['Params_M'])):
        pct = ((val - baseline_params) / baseline_params) * 100 if baseline_params > 0 else 0
        if df_params['Model'].iloc[i] == 'Hybrid':
            label = f"{val:.2f}M\n(baseline)"
        else:
            label = f"{val:.2f}M\n({pct:+.0f}%)"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 label, ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "efficiency_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"Efficiency chart saved to {VISUALIZATIONS_DIR}/efficiency_comparison.png")
    plt.close()
    return df

def analyze_probability_distribution(model, test_loader, device, class_names):
    """Analyze the probability distribution of predictions."""
    print("Analyzing Probability Distributions...")
    model.eval()
    
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            all_probs.append(probabilities.cpu().numpy())
            all_preds.append(probabilities.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    entropy = -np.sum(all_probs * np.log(all_probs + 1e-8), axis=1)
    max_probs = all_probs.max(axis=1)
    correct_mask = all_preds == all_labels
    
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Confidence Distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(max_probs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {max_probs.mean():.2f}')
    ax1.set_xlabel('Confidence')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    
    # 2. Entropy
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(entropy, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Entropy')
    ax2.set_title('Prediction Entropy (Lower=Better)')
    
    # 3. Accuracy by Class
    ax3 = plt.subplot(2, 3, 3)
    class_accs = []
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_accs.append((all_preds[mask] == all_labels[mask]).mean())
        else:
            class_accs.append(0)
    bars = ax3.bar(class_names, class_accs, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))), edgecolor='black')
    ax3.set_ylim(0, 1)
    ax3.set_title('Accuracy per Class')
    for bar, acc in zip(bars, class_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.0%}', ha='center')
    
    # 4. Summary Stats
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    stats_text = f"Overall Accuracy: {correct_mask.mean()*100:.1f}%\n"
    stats_text += f"Total Samples: {len(all_labels)}\n"
    stats_text += f"Mean Confidence: {max_probs.mean():.3f}\n"
    stats_text += f"Mean Entropy: {entropy.mean():.3f}"
    ax4.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=14, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Probability Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "probability_analysis.png"), dpi=150)
    plt.close()

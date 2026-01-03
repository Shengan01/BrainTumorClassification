import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools

def compare_models(models_stats):
    """
    Compare models using Fold Accuracies (Mann-Whitney U Test).
    models_stats: dict { 'ModelName': [acc_fold1, acc_fold2, ...] }
    """
    print("\nStatistical Comparison (Mann-Whitney U Test)")
    print("-" * 60)
    
    models = list(models_stats.keys())
    results = []
    
    for m1, m2 in itertools.combinations(models, 2):
        # Handle if stats are passed as direct list of accuracies (CV mode)
        # or as a dictionary of training history (Train mode, though train mode rarely compares)
        stats1 = models_stats[m1]
        stats2 = models_stats[m2]
        
        if isinstance(stats1, list):
            accs1 = stats1
        elif isinstance(stats1, dict) and 'val_acc' in stats1:
            accs1 = stats1['val_acc']
        else:
            print(f"Skipping {m1}: Unknown stats format")
            continue
            
        if isinstance(stats2, list):
            accs2 = stats2
        elif isinstance(stats2, dict) and 'val_acc' in stats2:
            accs2 = stats2['val_acc']
        else:
            print(f"Skipping {m2}: Unknown stats format")
            continue
        
        if len(accs1) < 2 or len(accs2) < 2:
            print(f"Skipping {m1}vs{m2}: Not enough folds ({len(accs1)}, {len(accs2)})")
            continue
            
        u_stat, p_val = stats.mannwhitneyu(accs1, accs2, alternative='two-sided')
        mean_diff = np.mean(accs1) - np.mean(accs2)
        
        results.append({
            'Model A': m1,
            'Model B': m2,
            'Mean Acc A': np.mean(accs1),
            'Mean Acc B': np.mean(accs2),
            'Diff': mean_diff,
            'P-Value': p_val,
            'Significant': p_val < 0.05
        })
        
        print(f"{m1} vs {m2}: Diff={mean_diff:.4f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")

    return pd.DataFrame(results)

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def calculate_metrics_per_class(y_true, y_pred, y_score, class_names):
    """
    Calculate comprehensive metrics including specificity and sensitivity per class.
    """
    n_classes = len(class_names)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Per-class specificity and sensitivity
    cm = confusion_matrix(y_true, y_pred)
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        # True Positives, False Negatives, False Positives, True Negatives
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[class_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1-score': report[class_name]['f1-score']
        }
    
    # AUC
    try:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        roc_auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr', average='macro')
    except:
        roc_auc = 0.0
        
    return report, roc_auc, metrics

def test_and_report(model, test_loader, device, class_names):
    """Evaluate model on test set and return comprehensive metrics."""
    from src.training.trainer import evaluate
    criterion = torch.nn.CrossEntropyLoss()
    _, acc, y_true, y_pred, y_score = evaluate(model, test_loader, criterion, device)
    
    y_score = np.array(y_score)
    report, roc_auc, per_class_metrics = calculate_metrics_per_class(y_true, y_pred, y_score, class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    return acc, report, roc_auc, cm, per_class_metrics

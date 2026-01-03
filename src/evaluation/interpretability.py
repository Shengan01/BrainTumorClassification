"""
Interpretability module using Captum library.
Includes SHAP, GradCAM, Occlusion, and Integrated Gradients.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import os
from src.config import VISUALIZATIONS_DIR, CLASS_NAMES, IMG_SIZE, CNN_VIZ_PERCENTILE, GRAD_VIZ_PERCENTILE

# Check if Captum is available
try:
    from captum.attr import LayerGradCam, IntegratedGradients, Occlusion, GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    print("Warning: Captum not installed. Run: pip install captum")
    CAPTUM_AVAILABLE = False


def _collect_correct_samples_per_class(model, test_loader, class_names, device, max_batches=200):
    """
    Collect one CORRECTLY PREDICTED sample per class.
    This ensures each visualization shows samples the model gets right.
    """
    model.eval()
    class_images = {i: None for i in range(len(class_names))}
    class_labels = {}
    
    batch_count = 0
    for images, labels in test_loader:
        batch_count += 1
        if batch_count > max_batches:
            break
        
        images_gpu = images.to(device)
        with torch.no_grad():
            preds = model(images_gpu).argmax(dim=1)
        
        for i in range(len(labels)):
            label = labels[i].item()
            pred = preds[i].item()
            
            # Only keep if correctly predicted AND we don't have this class yet
            if pred == label and class_images[label] is None:
                class_images[label] = images[i:i+1]
                class_labels[label] = label
        
        if all(v is not None for v in class_images.values()):
            break
    
    # Report any missing classes
    missing = [class_names[i] for i, v in class_images.items() if v is None]
    if missing:
        print(f"  Note: No correct predictions found for: {missing}")
        # Fall back to any sample for missing classes
        for images, labels in test_loader:
            for i in range(len(labels)):
                label = labels[i].item()
                if class_images[label] is None:
                    class_images[label] = images[i:i+1]
                    class_labels[label] = label
            if all(v is not None for v in class_images.values()):
                break
    
    sample_images = torch.cat([class_images[i] for i in range(len(class_names))], dim=0).to(device)
    sample_labels = [class_labels[i] for i in range(len(class_names))]
    return sample_images, sample_labels


def _normalize_and_smooth(attr, sigma=2):
    """Normalize attribution and apply Gaussian smoothing for better visualization."""
    attr = attr.detach().cpu().numpy()
    if len(attr.shape) == 4:
        attr = attr.squeeze(0)
    if len(attr.shape) == 3:
        attr = np.abs(attr).sum(axis=0)
    
    # Apply Gaussian smoothing
    attr = gaussian_filter(attr, sigma=sigma)
    
    # Normalize to [0, 1]
    attr = attr - attr.min()
    if attr.max() > 0:
        attr = attr / attr.max()
    return attr


def _prepare_image(img_tensor):
    """Prepare image tensor for display."""
    img_np = img_tensor.cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    return img_np


# ==============================================================================
# HYBRID-AWARE EXPLAINABILITY (CNN + Transformer fusion)
# ==============================================================================

def _get_cnn_feature_map(model, input_tensor, device, before_attention=False):
    """
    Extract CNN feature map from the tokenizer's last conv layer.
    
    Args:
        before_attention: If True, extract features BEFORE attention mechanisms
                         This produces cleaner GradCAMs as attention diffuses gradients
    """
    model.eval()
    feature_map = None
    
    def hook_fn(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()
    
    # Register hook on the appropriate layer
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
        
        if before_attention:
            # Hook BEFORE attention - get raw CNN features
            # This is layer3's output before attention is applied
            # We need to hook the layer itself, not after attention
            if hasattr(tokenizer, 'layer3'):
                hook = tokenizer.layer3.register_forward_hook(hook_fn)
            elif hasattr(tokenizer, 'layer2'):
                hook = tokenizer.layer2.register_forward_hook(hook_fn)
            else:
                return None
        else:
            # Hook the last CNN layer (standard behavior)
            if hasattr(tokenizer, 'layer4'):
                hook = tokenizer.layer4.register_forward_hook(hook_fn)
            elif hasattr(tokenizer, 'layer3'):
                hook = tokenizer.layer3.register_forward_hook(hook_fn)
            elif hasattr(tokenizer, 'layer2'):
                hook = tokenizer.layer2.register_forward_hook(hook_fn)
            else:
                return None
    else:
        return None
    
    try:
        with torch.no_grad():
            _ = model(input_tensor)
    finally:
        hook.remove()
    
    return feature_map


def _get_cnn_feature_before_attention(model, input_tensor, device):
    """
    Extract CNN features BEFORE attention mechanisms are applied.
    This produces cleaner, more localized activation maps.
    
    We run a partial forward through the tokenizer up to but not including attention.
    """
    model.eval()
    
    if not hasattr(model, 'tokenizer'):
        return None
    
    tokenizer = model.tokenizer
    
    with torch.no_grad():
        x = input_tensor
        
        # Run through CNN layers only (no attention)
        x = tokenizer.initial(x)
        
        if hasattr(tokenizer, 'layer1'):
            x = tokenizer.layer1(x)
            x = F.max_pool2d(x, 2)
        
        if hasattr(tokenizer, 'layer2'):
            x = tokenizer.layer2(x)
            x = F.max_pool2d(x, 2)
        
        if hasattr(tokenizer, 'layer3'):
            x = tokenizer.layer3(x)
            x = F.max_pool2d(x, 2)
        
        # Return features BEFORE attention and projection
        return x


def _compute_gradient_attention(model, input_tensor, target_class, device):
    """
    Compute gradient-based attention map for the entire model.
    This captures how both CNN and Transformer contribute to the decision.
    """
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    output = model(input_tensor)
    model.zero_grad()
    
    # Backward from the target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Get gradients w.r.t. input
    gradients = input_tensor.grad.detach()
    
    # Take absolute value and sum across channels
    attention = gradients.abs().sum(dim=1, keepdim=True)
    
    return attention


def _threshold_map(feature_map, percentile=CNN_VIZ_PERCENTILE):
    """
    Apply thresholding to only keep the top activations.
    Removes noise and highlights the strongest responses.
    """
    threshold = np.percentile(feature_map, percentile)
    feature_map_thresh = feature_map.copy()
    feature_map_thresh[feature_map_thresh < threshold] = 0
    return feature_map_thresh


def _compute_smoothgrad(model, input_tensor, target_class, device, n_samples=30, noise_level=0.15):
    """
    SmoothGrad: Average gradients over noisy versions of the input.
    Produces much smoother and more reliable saliency maps.
    """
    model.eval()
    
    # Get input stats for noise scaling
    std = input_tensor.std().item()
    noise_std = noise_level * std
    
    accumulated_grads = None
    
    for _ in range(n_samples):
        # Add Gaussian noise
        noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_std
        noisy_input = noisy_input.clone().requires_grad_(True)
        
        output = model(noisy_input)
        model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        grad = noisy_input.grad.detach().abs()
        
        if accumulated_grads is None:
            accumulated_grads = grad
        else:
            accumulated_grads += grad
    
    # Average
    smoothgrad = accumulated_grads / n_samples
    smoothgrad = smoothgrad.sum(dim=1, keepdim=True)
    
    return smoothgrad


def _extract_transformer_attention(model, input_tensor, device):
    """
    Extract transformer attention using Attention Rollout.
    
    Attention Rollout (Abnar & Zuidema, 2020) computes how attention flows
    through the transformer by multiplying attention matrices across layers.
    This produces much more interpretable visualizations than raw attention weights.
    
    Algorithm:
    1. Get attention weights from each layer
    2. Add residual connection (identity) to handle skip connections
    3. Re-normalize each row
    4. Multiply matrices across layers to get accumulated attention flow
    5. Extract CLS token's attention to spatial tokens
    """
    model.eval()
    
    if not hasattr(model, 'transformer'):
        return None
    
    try:
        # Forward pass to populate attention weights
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Get attention weights from transformer
        if not hasattr(model.transformer, 'get_attention_weights'):
            return None
        
        attention_weights = model.transformer.get_attention_weights()
        if attention_weights is None or len(attention_weights) == 0:
            return None
        
        # Stack all layers: list of (batch, seq_len, seq_len) -> (layers, batch, seq, seq)
        all_attn = torch.stack(attention_weights, dim=0)  # (num_layers, batch, seq, seq)
        
        # === ATTENTION ROLLOUT ===
        # Start with identity matrix (residual connections)
        num_layers = all_attn.shape[0]
        seq_len = all_attn.shape[2]
        
        # Initialize rollout with first layer's attention + residual
        # Add identity matrix to account for residual connections
        # Then renormalize rows to sum to 1
        rollout = all_attn[0, 0]  # (seq, seq) - first layer
        identity = torch.eye(seq_len, device=rollout.device)
        
        # Add residual connection and renormalize
        rollout = 0.5 * rollout + 0.5 * identity
        rollout = rollout / rollout.sum(dim=-1, keepdim=True)
        
        # Multiply through remaining layers
        for i in range(1, num_layers):
            attn_layer = all_attn[i, 0]  # (seq, seq)
            # Add residual connection
            attn_layer = 0.5 * attn_layer + 0.5 * identity
            attn_layer = attn_layer / attn_layer.sum(dim=-1, keepdim=True)
            # Matrix multiplication to accumulate attention flow
            rollout = torch.matmul(attn_layer, rollout)
        
        # Extract CLS token (index 0) attention to all spatial tokens
        # CLS attends to tokens 1:end (skip CLS self-attention)
        cls_attention = rollout[0, 1:]  # (seq_len - 1,)
        
        # Reshape to spatial dimensions (28x28 = 784 tokens for our hybrid)
        spatial_seq_len = cls_attention.shape[0]
        h = w = int(np.sqrt(spatial_seq_len))
        
        if h * w == spatial_seq_len:
            spatial_attn = cls_attention.view(1, 1, h, w)
            return spatial_attn
        
        # Try common sizes
        for possible_h, possible_w in [(28, 28), (14, 14), (7, 7), (56, 56)]:
            if possible_h * possible_w == spatial_seq_len:
                spatial_attn = cls_attention.view(1, 1, possible_h, possible_w)
                return spatial_attn
        
        return None
        
    except Exception as e:
        print(f"  Transformer attention extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def hybrid_attention_map(model, test_loader, device, class_names, model_name="Model"):
    """
    Generate Enhanced Hybrid Attention Maps with:
    1. CNN features (thresholded - top 25% only)
    2. SmoothGrad saliency (averaged over noisy inputs)
    3. Transformer attention (actual attention weights when available)
    """
    print(f"Generating Enhanced Hybrid Attention Map for {model_name}...")
    model = model.to(device).eval()
    
    sample_images, sample_labels = _collect_correct_samples_per_class(
        model, test_loader, class_names, device
    )
    
    num_classes = len(class_names)
    # 4 rows: Original, CNN (thresholded), SmoothGrad, Transformer Attention
    fig, axes = plt.subplots(4, num_classes, figsize=(num_classes*3, 12))
    fig.suptitle(f"Enhanced Attention Analysis - {model_name}", fontsize=14, fontweight='bold')
    
    import cv2
    
    for i in range(num_classes):
        input_tensor = sample_images[i:i+1].to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        
        # 1. Get CNN feature map BEFORE attention (cleaner visualization)
        cnn_features = _get_cnn_feature_before_attention(model, input_tensor, device)
        
        # 2. Get SmoothGrad saliency (smoother than regular gradients)
        smoothgrad = _compute_smoothgrad(model, input_tensor, pred, device, n_samples=10)
        
        # 3. Get transformer attention (if available)
        transformer_attn = _extract_transformer_attention(model, input_tensor, device)
        
        # Prepare original image
        img_np = _prepare_image(sample_images[i])
        
        # Row 0: Original image
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True: {class_names[sample_labels[i]]}")
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)
        
        # Row 1: CNN Feature activation (THRESHOLDED - top 25% only)
        if cnn_features is not None:
            cnn_map = cnn_features[0].sum(dim=0).cpu().numpy()
            cnn_map = (cnn_map - cnn_map.min()) / (cnn_map.max() - cnn_map.min() + 1e-8)
            # Apply thresholding - keep only top 25%
            cnn_map = _threshold_map(cnn_map, percentile=CNN_VIZ_PERCENTILE)
            cnn_map = gaussian_filter(cnn_map, sigma=1)
            cnn_map_resized = cv2.resize(cnn_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            
            axes[1, i].imshow(img_np)
            axes[1, i].imshow(cnn_map_resized, cmap='jet', alpha=0.6)
            axes[1, i].set_title(f"CNN (Top {100-CNN_VIZ_PERCENTILE}%)")
            axes[1, i].axis('off')
        else:
            axes[1, i].imshow(img_np)
            axes[1, i].set_title("No CNN layer")
            axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel("CNN Features", fontsize=10)
        
        # Row 2: SmoothGrad saliency
        if smoothgrad is not None:
            grad_map = smoothgrad[0, 0].cpu().numpy()
            grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
            grad_map = _threshold_map(grad_map, percentile=GRAD_VIZ_PERCENTILE)
            grad_map = gaussian_filter(grad_map, sigma=1)
            
            axes[2, i].imshow(img_np)
            axes[2, i].imshow(grad_map, cmap='jet', alpha=0.6)
            axes[2, i].set_title(f"Pred: {class_names[pred]}")
            axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel("SmoothGrad", fontsize=10)
        
        # Row 3: Transformer Attention (if available)
        if transformer_attn is not None:
            attn_map = transformer_attn[0, 0].cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_map = _threshold_map(attn_map, percentile=GRAD_VIZ_PERCENTILE)
            attn_map_resized = cv2.resize(attn_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            
            axes[3, i].imshow(img_np)
            axes[3, i].imshow(attn_map_resized, cmap='jet', alpha=0.6)
            axes[3, i].set_title("Transformer")
            axes[3, i].axis('off')
        else:
            axes[3, i].imshow(img_np)
            axes[3, i].set_title("No Transformer")
            axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel("Attn Weights", fontsize=10)
    
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    save_path = os.path.join(VISUALIZATIONS_DIR, f"hybrid_attention_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hybrid attention map saved to {save_path}")


def hybrid_attention_compact(model, test_loader, device, class_names, model_name="Model"):
    """
    Generate a compact single-row visualization showing SmoothGrad saliency.
    Ideal for papers and presentations.
    """
    print(f"Generating Compact SmoothGrad for {model_name}...")
    model = model.to(device).eval()
    
    sample_images, sample_labels = _collect_correct_samples_per_class(
        model, test_loader, class_names, device
    )
    
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes*4, 4))
    fig.suptitle(f"SmoothGrad Saliency - {model_name}", fontsize=14, fontweight='bold')
    
    for i in range(num_classes):
        input_tensor = sample_images[i:i+1].to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        
        # Use SmoothGrad instead of regular gradients
        smoothgrad = _compute_smoothgrad(model, input_tensor, pred, device, n_samples=10)
        img_np = _prepare_image(sample_images[i])
        
        axes[i].imshow(img_np)
        
        if smoothgrad is not None:
            grad_map = smoothgrad[0, 0].cpu().numpy()
            grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
            grad_map = _threshold_map(grad_map, percentile=GRAD_VIZ_PERCENTILE)
            grad_map = gaussian_filter(grad_map, sigma=1)
            axes[i].imshow(grad_map, cmap='jet', alpha=0.6)
        
        # Show both true label and prediction
        correct = "✓" if sample_labels[i] == pred else "✗"
        axes[i].set_title(f"{class_names[sample_labels[i]]} {correct}", fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    save_path = os.path.join(VISUALIZATIONS_DIR, f"hybrid_compact_{safe_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Compact hybrid attention saved to {save_path}")


def visualize_ablation_comparison(test_loader, device, class_names, experiments_dir):
    """
    Generate SEPARATE visualization images for each ablation model.
    Each ablation gets its own image showing attention patterns across all classes.
    """
    from src.models.hybrid import HybridTumorClassifier
    from src.models.ablation import get_ablation_model, ABLATION_NAMES
    
    print("Generating Individual Ablation Visualizations...")
    
    # Collect one sample per class using the full Hybrid model first
    full_model = HybridTumorClassifier(num_classes=4).to(device)
    full_path = os.path.join(experiments_dir, "hybrid_model.pth")
    if os.path.exists(full_path):
        full_model.load_state_dict(torch.load(full_path, map_location=device))
    
    sample_images, sample_labels = _collect_correct_samples_per_class(
        full_model, test_loader, class_names, device
    )
    
    # Model configurations to generate
    ablation_file_map = {
        "Without Channel Attention": "without_channel_attention",
        "Without Spatial Attention": "without_spatial_attention", 
        "Without Transformer": "without_transformer",
        "Without Attention Pooling": "without_attention_pooling",
        "Without Dropout": "without_dropout",
        "Reduced Hidden Dim": "reduced_hidden_dim",
        "Without All Attention": "without_all_attention"
    }
    
    # Generate image for Full Hybrid first
    _generate_single_ablation_image(
        full_model, "Full Hybrid", sample_images, sample_labels, 
        class_names, device, "full_hybrid"
    )
    _generate_compact_ablation_image(
        full_model, "Full Hybrid", sample_images, sample_labels, 
        class_names, device, "full_hybrid"
    )
    
    # Generate image for each ablation
    for name in ABLATION_NAMES:
        safe_name = ablation_file_map.get(name, name.lower().replace(" ", "_"))
        # Prioritize _model.pth (best fold from CV) over .pth (last fold from trainer)
        path = os.path.join(experiments_dir, f"{safe_name}_model.pth")
        if not os.path.exists(path):
            path = os.path.join(experiments_dir, f"{safe_name}.pth")
        
        if os.path.exists(path):
            try:
                model = get_ablation_model(name, num_classes=4).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                _generate_single_ablation_image(
                    model, name, sample_images, sample_labels, 
                    class_names, device, safe_name
                )
                _generate_compact_ablation_image(
                    model, name, sample_images, sample_labels, 
                    class_names, device, safe_name
                )
                print(f"  Generated: {name}")
            except RuntimeError as e:
                print(f"  Skipping {name}: incompatible weights")
            finally:
                if 'model' in dir():
                    del model
                torch.cuda.empty_cache()
        else:
            print(f"  Not found: {name} ({path})")
    
    # Also generate the combined comparison grid
    _generate_ablation_comparison_grid(
        test_loader, device, class_names, experiments_dir, 
        sample_images, sample_labels
    )
    
    # Clean up
    del full_model
    torch.cuda.empty_cache()


def _generate_single_ablation_image(model, model_name, sample_images, sample_labels, 
                                     class_names, device, safe_name):
    """
    Generate enhanced visualization for one ablation model:
    - Row 1: Original images
    - Row 2: CNN feature activation (thresholded)
    - Row 3: SmoothGrad saliency
    """
    model = model.to(device).eval()
    num_classes = len(class_names)
    
    import cv2
    
    fig, axes = plt.subplots(3, num_classes, figsize=(num_classes*3, 9))
    fig.suptitle(f"Enhanced Analysis - {model_name}", fontsize=14, fontweight='bold')
    
    for i in range(num_classes):
        input_tensor = sample_images[i:i+1].to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        
        # Get CNN feature map BEFORE attention (cleaner visualization)
        cnn_features = _get_cnn_feature_before_attention(model, input_tensor, device)
        
        # Get SmoothGrad saliency (instead of regular gradients)
        smoothgrad = _compute_smoothgrad(model, input_tensor, pred, device, n_samples=10)
        
        # Prepare original image
        img_np = _prepare_image(sample_images[i])
        
        # Row 0: Original image
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True: {class_names[sample_labels[i]]}")
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)
        
        # Row 1: CNN Feature activation (THRESHOLDED)
        if cnn_features is not None:
            cnn_map = cnn_features[0].sum(dim=0).cpu().numpy()
            cnn_map = (cnn_map - cnn_map.min()) / (cnn_map.max() - cnn_map.min() + 1e-8)
            cnn_map = _threshold_map(cnn_map, percentile=CNN_VIZ_PERCENTILE)
            cnn_map = gaussian_filter(cnn_map, sigma=1)
            cnn_map_resized = cv2.resize(cnn_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            
            axes[1, i].imshow(img_np)
            axes[1, i].imshow(cnn_map_resized, cmap='jet', alpha=0.6)
            axes[1, i].set_title(f"CNN (Top {100-CNN_VIZ_PERCENTILE}%)")
            axes[1, i].axis('off')
        else:
            axes[1, i].imshow(img_np)
            axes[1, i].set_title("No CNN layer")
            axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel("CNN Features", fontsize=10)
        
        # Row 2: SmoothGrad saliency
        if smoothgrad is not None:
            grad_map = smoothgrad[0, 0].cpu().numpy()
            grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
            grad_map = _threshold_map(grad_map, percentile=GRAD_VIZ_PERCENTILE)
            grad_map = gaussian_filter(grad_map, sigma=1)
            
            axes[2, i].imshow(img_np)
            axes[2, i].imshow(grad_map, cmap='jet', alpha=0.6)
            axes[2, i].set_title(f"Pred: {class_names[pred]}")
            axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel("SmoothGrad", fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, f"hybrid_attention_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _generate_compact_ablation_image(model, model_name, sample_images, sample_labels, 
                                      class_names, device, safe_name):
    """
    Generate a compact single-row visualization using SmoothGrad saliency.
    """
    model = model.to(device).eval()
    num_classes = len(class_names)
    
    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes*4, 4))
    fig.suptitle(f"SmoothGrad Saliency - {model_name}", fontsize=14, fontweight='bold')
    
    for i in range(num_classes):
        input_tensor = sample_images[i:i+1].to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        
        # Use SmoothGrad
        smoothgrad = _compute_smoothgrad(model, input_tensor, pred, device, n_samples=10)
        img_np = _prepare_image(sample_images[i])
        
        axes[i].imshow(img_np)
        
        if smoothgrad is not None:
            grad_map = smoothgrad[0, 0].cpu().numpy()
            grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
            grad_map = _threshold_map(grad_map, percentile=GRAD_VIZ_PERCENTILE)
            grad_map = gaussian_filter(grad_map, sigma=1)
            axes[i].imshow(grad_map, cmap='jet', alpha=0.6)
        
        correct = "✓" if sample_labels[i] == pred else "✗"
        axes[i].set_title(f"{class_names[sample_labels[i]]} {correct}", fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, f"hybrid_compact_{safe_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def _generate_ablation_comparison_grid(test_loader, device, class_names, experiments_dir,
                                        sample_images, sample_labels):
    """Generate the combined comparison grid (all ablations in one image)."""
    from src.models.hybrid import HybridTumorClassifier
    from src.models.ablation import get_ablation_model, ABLATION_NAMES
    
    full_model = HybridTumorClassifier(num_classes=4).to(device)
    full_path = os.path.join(experiments_dir, "hybrid_model.pth")
    if os.path.exists(full_path):
        full_model.load_state_dict(torch.load(full_path, map_location=device))
    
    model_configs = [("Full Hybrid", full_model, full_path)]
    
    ablation_file_map = {
        "Without Spatial Attention": "without_spatial_attention", 
        "Without Transformer": "without_transformer",
        "Without Attention Pooling": "without_attention_pooling",
        "Without Dropout": "without_dropout",
        "Without Channel Attention": "without_channel_attention",
        "Reduced Hidden Dim": "reduced_hidden_dim",
        "Without All Attention": "without_all_attention"
    }
    
    for name in ABLATION_NAMES:
        safe_name = ablation_file_map.get(name, name.lower().replace(" ", "_"))
        # Prioritize _model.pth (best fold from CV) over .pth (last fold from trainer)
        path = os.path.join(experiments_dir, f"{safe_name}_model.pth")
        if not os.path.exists(path):
            path = os.path.join(experiments_dir, f"{safe_name}.pth")
        
        if os.path.exists(path):
            try:
                model = get_ablation_model(name, num_classes=4).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model_configs.append((name, model, path))
            except RuntimeError:
                print(f"  Skipping {name}: incompatible weights")
    
    if len(model_configs) < 2:
        return
    
    n_models = len(model_configs)
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(n_models, n_classes, figsize=(n_classes*3, n_models*2.5))
    fig.suptitle("Ablation Attention Comparison", fontsize=14, fontweight='bold', y=1.02)
    
    for row, (model_name, model, _) in enumerate(model_configs):
        model = model.to(device).eval()
        
        for col in range(n_classes):
            input_tensor = sample_images[col:col+1].to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
            
            grad_attention = _compute_gradient_attention(model, input_tensor, pred, device)
            img_np = _prepare_image(sample_images[col])
            
            ax = axes[row, col] if n_models > 1 else axes[col]
            ax.imshow(img_np)
            
            if grad_attention is not None:
                grad_map = grad_attention[0, 0].cpu().numpy()
                grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
                grad_map = gaussian_filter(grad_map, sigma=2)
                ax.imshow(grad_map, cmap='jet', alpha=0.5)
            
            if row == 0:
                ax.set_title(class_names[col], fontsize=10)
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(model_name, fontsize=9, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, "ablation_attention_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined comparison saved to {save_path}")
    
    # Clean up
    del full_model
    torch.cuda.empty_cache()


def analyze_with_shap(model, test_loader, device, class_names, model_name="Model", n_samples=10):
    """
    SHAP values using Captum's GradientShap.
    More reliable than shap library for neural networks.
    """
    if not CAPTUM_AVAILABLE:
        print("Captum not available")
        return
    
    print(f"Running SHAP (GradientShap) for {model_name}...")
    model = model.to(device).eval()
    
    sample_images, sample_labels = _collect_correct_samples_per_class(model, test_loader, class_names, device)
    
    try:
        # Use a subset of images as baseline distribution
        baseline_images = []
        count = 0
        for images, _ in test_loader:
            baseline_images.append(images)
            count += images.shape[0]
            if count >= n_samples:
                break
        baseline = torch.cat(baseline_images, dim=0)[:n_samples].to(device)
        
        gs = GradientShap(model)
        
        num_classes = len(class_names)
        fig, axes = plt.subplots(2, num_classes, figsize=(num_classes*3, 6))
        fig.suptitle(f"SHAP Values - {model_name}", fontsize=14, fontweight='bold')
        
        for i in range(num_classes):
            input_tensor = sample_images[i:i+1]
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
            
            # Compute SHAP values
            attribution = gs.attribute(input_tensor, baselines=baseline, target=pred, n_samples=5)
            attr_np = _normalize_and_smooth(attribution, sigma=3)
            img_np = _prepare_image(sample_images[i])
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"True: {class_names[sample_labels[i]]}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img_np)
            axes[1, i].imshow(attr_np, cmap='jet', alpha=0.5)
            axes[1, i].set_title(f"Pred: {class_names[pred]}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"shap_{safe_name}.png"), dpi=150)
        plt.close()
        print(f"SHAP saved to {VISUALIZATIONS_DIR}/shap_{safe_name}.png")
        
    except Exception as e:
        print(f"SHAP error for {model_name}: {e}")
        import traceback
        traceback.print_exc()


def gradcam_captum(model, test_loader, device, class_names, model_name="Model", target_layer=None):
    """GradCAM with smooth overlay."""
    if not CAPTUM_AVAILABLE:
        print("Captum not available")
        return
    
    print(f"Running GradCAM for {model_name}...")
    model = model.to(device).eval()
    
    if target_layer is None:
        print(f"No target layer specified for {model_name}")
        return
    
    sample_images, sample_labels = _collect_correct_samples_per_class(model, test_loader, class_names, device)
    
    try:
        layer_gc = LayerGradCam(model, target_layer)
        
        num_classes = len(class_names)
        fig, axes = plt.subplots(2, num_classes, figsize=(num_classes*3, 6))
        fig.suptitle(f"GradCAM - {model_name}", fontsize=14, fontweight='bold')
        
        for i in range(num_classes):
            input_tensor = sample_images[i:i+1]
            input_tensor.requires_grad = True
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
            
            attribution = layer_gc.attribute(input_tensor, target=pred)
            attribution = torch.nn.functional.interpolate(
                attribution, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
            )
            attr_np = _normalize_and_smooth(attribution, sigma=1)
            img_np = _prepare_image(sample_images[i])
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"True: {class_names[sample_labels[i]]}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img_np)
            axes[1, i].imshow(attr_np, cmap='jet', alpha=0.5)
            axes[1, i].set_title(f"Pred: {class_names[pred]}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"gradcam_{safe_name}.png"), dpi=150)
        plt.close()
        print(f"GradCAM saved to {VISUALIZATIONS_DIR}/gradcam_{safe_name}.png")
        
    except Exception as e:
        print(f"GradCAM error for {model_name}: {e}")
        import traceback
        traceback.print_exc()


def occlusion_sensitivity(model, test_loader, device, class_names, model_name="Model"):
    """Occlusion Sensitivity with smaller windows and smoothing."""
    if not CAPTUM_AVAILABLE:
        print("Captum not available")
        return
    
    print(f"Running Occlusion for {model_name}...")
    model = model.to(device).eval()
    
    sample_images, sample_labels = _collect_correct_samples_per_class(model, test_loader, class_names, device)
    
    try:
        occlusion = Occlusion(model)
        
        num_classes = len(class_names)
        fig, axes = plt.subplots(2, num_classes, figsize=(num_classes*3, 6))
        fig.suptitle(f"Occlusion Sensitivity - {model_name}", fontsize=14, fontweight='bold')
        
        for i in range(num_classes):
            input_tensor = sample_images[i:i+1]
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
            
            n_channels = input_tensor.shape[1]
            # Smaller window (8x8) and stride (4) for smoother result
            attribution = occlusion.attribute(
                input_tensor,
                target=pred,
                strides=(n_channels, 4, 4),
                sliding_window_shapes=(n_channels, 8, 8),
                baselines=0
            )
            
            attr_np = _normalize_and_smooth(attribution, sigma=4)
            img_np = _prepare_image(sample_images[i])
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"True: {class_names[sample_labels[i]]}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img_np)
            axes[1, i].imshow(attr_np, cmap='jet', alpha=0.5)
            axes[1, i].set_title(f"Pred: {class_names[pred]}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"occlusion_{safe_name}.png"), dpi=150)
        plt.close()
        print(f"Occlusion saved to {VISUALIZATIONS_DIR}/occlusion_{safe_name}.png")
        
    except Exception as e:
        print(f"Occlusion error for {model_name}: {e}")
        import traceback
        traceback.print_exc()


def integrated_gradients_captum(model, test_loader, device, class_names, model_name="Model", n_steps=50):
    """Integrated Gradients with smoothing."""
    if not CAPTUM_AVAILABLE:
        print("Captum not available")
        return
    
    print(f"Running Integrated Gradients for {model_name}...")
    model = model.to(device).eval()
    
    sample_images, sample_labels = _collect_correct_samples_per_class(model, test_loader, class_names, device)
    
    try:
        ig = IntegratedGradients(model)
        
        num_classes = len(class_names)
        fig, axes = plt.subplots(2, num_classes, figsize=(num_classes*3, 6))
        fig.suptitle(f"Integrated Gradients - {model_name}", fontsize=14, fontweight='bold')
        
        for i in range(num_classes):
            input_tensor = sample_images[i:i+1]
            baseline = torch.zeros_like(input_tensor)
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
            
            attribution = ig.attribute(input_tensor, baselines=baseline, target=pred, n_steps=n_steps)
            attr_np = _normalize_and_smooth(attribution, sigma=3)
            img_np = _prepare_image(sample_images[i])
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"True: {class_names[sample_labels[i]]}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img_np)
            axes[1, i].imshow(attr_np, cmap='jet', alpha=0.5)
            axes[1, i].set_title(f"Pred: {class_names[pred]}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"integrated_gradients_{safe_name}.png"), dpi=150)
        plt.close()
        print(f"Integrated Gradients saved to {VISUALIZATIONS_DIR}/integrated_gradients_{safe_name}.png")
        
    except Exception as e:
        print(f"Integrated Gradients error for {model_name}: {e}")
        import traceback
        traceback.print_exc()


# Legacy compatibility
def improved_gradcam_visualization(model, test_loader, device, class_names, num_samples=8):
    """Legacy GradCAM fallback."""
    integrated_gradients_captum(model, test_loader, device, class_names, model_name="Hybrid")

# Aliases
gradcam_for_cnn = gradcam_captum
attention_visualization_vit = integrated_gradients_captum
integrated_gradients_visualization = integrated_gradients_captum

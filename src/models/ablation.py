from src.models.hybrid import HybridTumorClassifier, TinyHybrid

def get_ablation_model(name, num_classes=4):
    """
    Factory function to get ablation models for both Hybrid and TinyHybrid.
    
    Hybrid ablations test the full 0.88 GFLOP model.
    TinyHybrid ablations test the ultra-efficient 0.05 GFLOP model.
    """
    # ===== HYBRID ABLATIONS =====
    if name == "Hybrid: No Channel Attn":
        return HybridTumorClassifier(num_classes=num_classes, use_channel_attn=False)
    
    elif name == "Hybrid: No Spatial Attn":
        return HybridTumorClassifier(num_classes=num_classes, use_spatial_attn=False)
    
    elif name == "Hybrid: No Transformer":
        return HybridTumorClassifier(num_classes=num_classes, use_transformer=False)
    
    elif name == "Hybrid: No Attn Pooling":
        return HybridTumorClassifier(num_classes=num_classes, use_attn_pooling=False)
    
    elif name == "Hybrid: No Dropout":
        return HybridTumorClassifier(num_classes=num_classes, use_dropout=False)
    
    elif name == "Hybrid: No All Attn":
        return HybridTumorClassifier(
            num_classes=num_classes, 
            use_channel_attn=False, 
            use_spatial_attn=False, 
            use_attn_pooling=False
        )
    
    elif name == "Hybrid: Reduced CNN Layers":
        # 2 CNN layers instead of 3 - tests impact of CNN depth
        return HybridTumorClassifier(num_classes=num_classes, num_cnn_layers=2)
    
    elif name == "Hybrid: Optimized":
        # Best combo from ablation study: No Spatial Attn + No Dropout
        # No Spatial Attn gave +0.77%, No Dropout gave +0.16%
        return HybridTumorClassifier(
            num_classes=num_classes,
            use_spatial_attn=False,
            use_dropout=False
        )
    
    elif name == "Hybrid: Simple DSC":
        # Simple Depthwise-Separable Conv (no residual connection)
        # MobileNetV1 style - ~50% less compute in CNN backbone
        return HybridTumorClassifier(num_classes=num_classes, block_type='simple')
    
    elif name == "Hybrid: Inverted Residual":
        # Inverted Residual Block (MobileNetV2 style)
        # Better gradient flow, more efficient than standard residual
        return HybridTumorClassifier(num_classes=num_classes, block_type='inverted')
    
    # ===== TINYHYBRID ABLATIONS =====
    elif name == "Tiny: No Channel Attn":
        return TinyHybrid(num_classes=num_classes, use_channel_attn=False)
    
    elif name == "Tiny: No Spatial Attn":
        return TinyHybrid(num_classes=num_classes, use_spatial_attn=False)
    
    elif name == "Tiny: No Transformer":
        return TinyHybrid(num_classes=num_classes, use_transformer=False)
    
    elif name == "Tiny: No Attn Pooling":
        return TinyHybrid(num_classes=num_classes, use_attn_pooling=False)
    
    elif name == "Tiny: No Dropout":
        return TinyHybrid(num_classes=num_classes, use_dropout=False)
    
    elif name == "Tiny: No All Attn":
        return TinyHybrid(
            num_classes=num_classes, 
            use_channel_attn=False, 
            use_spatial_attn=False, 
            use_attn_pooling=False
        )
    
    elif name == "Tiny: Reduced CNN Layers":
        # 2 CNN layers instead of 3 - tests impact of CNN depth
        return TinyHybrid(num_classes=num_classes, num_cnn_layers=2)
    
    elif name == "Tiny: Optimized":
        # Same pattern as Hybrid: No Spatial Attn + No Dropout
        return TinyHybrid(
            num_classes=num_classes,
            use_spatial_attn=False,
            use_dropout=False
        )
    
    elif name == "Tiny: Simple DSC":
        # Simple Depthwise-Separable Conv (no residual connection)
        # MobileNetV1 style - ~50% less compute in CNN backbone
        return TinyHybrid(num_classes=num_classes, block_type='simple')
    
    elif name == "Tiny: Inverted Residual":
        # Inverted Residual Block (MobileNetV2 style)
        # Better gradient flow, more efficient than standard residual
        return TinyHybrid(num_classes=num_classes, block_type='inverted')
    
    else:
        raise ValueError(f"Unknown ablation: {name}")

# Ablation names organized by model type
HYBRID_ABLATION_NAMES = [
    "Hybrid: No Channel Attn",
    "Hybrid: No Spatial Attn",
    "Hybrid: No Transformer",
    "Hybrid: No Attn Pooling",
    "Hybrid: No Dropout",
    "Hybrid: No All Attn",
    "Hybrid: Reduced CNN Layers",
    "Hybrid: Optimized",
    "Hybrid: Simple DSC",
    "Hybrid: Inverted Residual",
]

TINY_ABLATION_NAMES = [
    "Tiny: No Channel Attn",
    "Tiny: No Spatial Attn",
    "Tiny: No Transformer",
    "Tiny: No Attn Pooling",
    "Tiny: No Dropout",
    "Tiny: No All Attn",
    "Tiny: Reduced CNN Layers",
    "Tiny: Optimized",
    "Tiny: Simple DSC",
    "Tiny: Inverted Residual",
]

# Block-type ablations only (for focused training)
BLOCK_TYPE_ABLATION_NAMES = [
    "Hybrid: Simple DSC",
    "Hybrid: Inverted Residual",
    "Tiny: Simple DSC",
    "Tiny: Inverted Residual",
]

# All ablation names combined
ABLATION_NAMES = HYBRID_ABLATION_NAMES + TINY_ABLATION_NAMES


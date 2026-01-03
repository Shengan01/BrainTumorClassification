import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import ChannelAttention, SpatialAttention, TransformerEncoder, AttentionPooling


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution - MobileNet style.
    Much more efficient than standard convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class EfficientResidualBlock(nn.Module):
    """
    Residual block using depthwise separable convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.relu(out)


class SimpleDepthwiseSeparable(nn.Module):
    """
    Simple Depthwise Separable Convolution block (MobileNetV1 style).
    No residual connection - just depthwise + pointwise + BN + ReLU.
    Approximately 50% less compute than EfficientResidualBlock.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 style).
    Expand -> Depthwise -> Project with residual on narrow ends.
    Better gradient flow than simple DSC, more efficient than standard residual.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand (only if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Project (linear - no activation!)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Shortcut for dimension mismatch
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x) + self.shortcut(x)


def get_block(block_type):
    """Factory function to get block class by name."""
    blocks = {
        'residual': EfficientResidualBlock,
        'simple': SimpleDepthwiseSeparable,
        'inverted': InvertedResidualBlock,
    }
    if block_type not in blocks:
        raise ValueError(f"Unknown block_type: {block_type}. Choose from {list(blocks.keys())}")
    return blocks[block_type]


class CNNTokenizer(nn.Module):
    """
    Efficient CNN tokenizer:
    - 2-3 stages with depthwise separable convolutions
    - 28x28 feature map (784 tokens) regardless of num_cnn_layers
    - Channel and spatial attention
    
    Args:
        num_cnn_layers: Number of CNN stages (2 or 3). Default 3.
            - 3 layers: 8->16->32 channels (deeper features)
            - 2 layers: 8->16 channels (shallower, tests CNN depth impact)
        block_type: Type of convolutional block ('residual', 'simple', 'inverted'). Default 'residual'.
    """
    def __init__(self, in_channels=1, hidden_dim=256, use_channel_attn=True, use_spatial_attn=True, 
                 num_cnn_layers=3, block_type='residual'):
        super().__init__()
        self.num_cnn_layers = num_cnn_layers
        
        # Get the block class based on block_type
        Block = get_block(block_type)
        
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Convolutional Stages (using selected block type)
        self.layer1 = Block(8, 8)
        self.layer2 = Block(8, 16)
        
        # Layer 3 is optional (for ablation study)
        if num_cnn_layers >= 3:
            self.layer3 = Block(16, 32)
            final_channels = 32
        else:
            self.layer3 = None
            final_channels = 16
        
        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn
        
        if use_channel_attn:
            self.channel_attention = ChannelAttention(final_channels)
        if use_spatial_attn:
            self.spatial_attention = SpatialAttention(final_channels)
            
        self.project = nn.Conv2d(final_channels, hidden_dim, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)  # 224x224
        
        x = self.layer1(x)
        x = F.max_pool2d(x, 2)  # 112x112
        
        x = self.layer2(x)
        x = F.max_pool2d(x, 2)  # 56x56
        
        if self.layer3 is not None:
            x = self.layer3(x)
            x = F.max_pool2d(x, 2)  # 28x28 feature map (784 tokens)
        else:
            # Extra pooling for 2-layer variant to maintain similar token count
            x = F.max_pool2d(x, 2)  # 28x28 feature map (784 tokens)
        
        if self.use_channel_attn:
            x = self.channel_attention(x)
        if self.use_spatial_attn:
            x = self.spatial_attention(x)
            
        x = self.project(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, N, hidden_dim)


class HybridTumorClassifier(nn.Module):
    """
    Efficient Hybrid CNN+Transformer:
    
    Architecture:
    - CNN: 2-3 stage efficient tokenizer (depthwise separable)
    - Transformer: 4-layer encoder (4 heads, 256 dim, 512 MLP)
    - ~0.88 GFLOPs, ~1.1M params (with 3 CNN layers)
    
    Args:
        block_type: Type of convolutional block ('residual', 'simple', 'inverted'). Default 'residual'.
    """
    def __init__(self, in_channels=1, hidden_dim=256, num_classes=4, 
                 use_channel_attn=True, use_spatial_attn=True, 
                 use_transformer=True, use_attn_pooling=True, use_dropout=True,
                 num_cnn_layers=3, block_type='residual'):
        super().__init__()
        
        self.tokenizer = CNNTokenizer(
            in_channels=in_channels, 
            hidden_dim=hidden_dim, 
            use_channel_attn=use_channel_attn, 
            use_spatial_attn=use_spatial_attn,
            num_cnn_layers=num_cnn_layers,
            block_type=block_type
        )
        
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerEncoder(
                dim=hidden_dim, 
                depth=4, 
                heads=4, 
                mlp_dim=512
            )
            
        self.use_attn_pooling = use_attn_pooling
        if use_attn_pooling:
            self.attn_pool = AttentionPooling(hidden_dim)
        
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.2)
            
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.tokenizer(x)  # (B, 784, hidden_dim)
        
        if self.use_transformer:
            x = self.transformer(x)
            
        if self.use_attn_pooling:
            x = self.attn_pool(x)
        else:
            x = x.mean(dim=1)
            
        if self.use_dropout:
            x = self.dropout(x)
            
        return self.head(x)


class TinyCNNTokenizer(nn.Module):
    """
    Ultra-compact CNN tokenizer for TinyHybrid:
    - 2-3 stages with depthwise separable convolutions
    - 14x14 or 28x28 feature map depending on num_cnn_layers
    - Optional channel and spatial attention
    
    Args:
        num_cnn_layers: Number of CNN stages (2 or 3). Default 3.
            - 3 layers: 8->16->24 channels, 14x14 feature map (196 tokens)
            - 2 layers: 8->16 channels, 28x28 feature map (784 tokens)
        block_type: Type of convolutional block ('residual', 'simple', 'inverted'). Default 'residual'.
    """
    def __init__(self, in_channels=1, hidden_dim=64, use_channel_attn=True, use_spatial_attn=True, 
                 num_cnn_layers=3, block_type='residual'):
        super().__init__()
        self.num_cnn_layers = num_cnn_layers
        
        # Get the block class based on block_type
        Block = get_block(block_type)
        
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Convolutional Stages (using selected block type)
        self.layer1 = Block(8, 8)
        self.layer2 = Block(8, 16)
        
        # Layer 3 is optional (for ablation study)
        if num_cnn_layers >= 3:
            self.layer3 = Block(16, 24)
            final_channels = 24
        else:
            self.layer3 = None
            final_channels = 16
        
        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn
        
        if use_channel_attn:
            self.channel_attention = ChannelAttention(final_channels)
        if use_spatial_attn:
            self.spatial_attention = SpatialAttention(final_channels)
            
        self.project = nn.Conv2d(final_channels, hidden_dim, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)  # 224x224
        
        x = self.layer1(x)
        x = F.max_pool2d(x, 2)  # 112x112
        
        x = self.layer2(x)
        x = F.max_pool2d(x, 2)  # 56x56
        
        if self.layer3 is not None:
            x = self.layer3(x)
            x = F.max_pool2d(x, 4)  # 14x14 feature map (196 tokens)
        else:
            # Extra pooling for 2-layer variant to maintain similar token count
            x = F.max_pool2d(x, 4)  # 14x14 feature map (196 tokens)
        
        if self.use_channel_attn:
            x = self.channel_attention(x)
        if self.use_spatial_attn:
            x = self.spatial_attention(x)
        
        x = self.project(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, N, hidden_dim)


class TinyHybrid(nn.Module):
    """
    Ultra-efficient Hybrid CNN+Transformer with ALL features enabled.
    
    Architecture:
    - CNN: 3-stage tiny tokenizer (8->16->24, depthwise separable, 14x14)
    - Channel + Spatial Attention in CNN
    - Transformer: 2-layer encoder (2 heads, 64 dim, 128 MLP)
    - Attention Pooling
    - Dropout (0.1)
    - ~0.05 GFLOPs, ~0.04M params
    
    All features can be disabled for ablation studies.
    
    Args:
        block_type: Type of convolutional block ('residual', 'simple', 'inverted'). Default 'residual'.
    """
    def __init__(self, in_channels=1, num_classes=4, hidden_dim=64,
                 use_channel_attn=True, use_spatial_attn=True,
                 use_transformer=True, use_attn_pooling=True, use_dropout=True,
                 num_cnn_layers=3, block_type='residual'):
        super().__init__()
        
        self.tokenizer = TinyCNNTokenizer(
            in_channels=in_channels, 
            hidden_dim=hidden_dim,
            use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn,
            num_cnn_layers=num_cnn_layers,
            block_type=block_type
        )
        
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerEncoder(
                dim=hidden_dim, 
                depth=2,
                heads=2,
                mlp_dim=128,
                max_seq_len=256
            )
        
        self.use_attn_pooling = use_attn_pooling
        if use_attn_pooling:
            self.attn_pool = AttentionPooling(hidden_dim)
        
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.2)
            
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.tokenizer(x)  # (B, 196, 64)
        
        if self.use_transformer:
            x = self.transformer(x)
        
        if self.use_attn_pooling:
            x = self.attn_pool(x)
        else:
            x = x.mean(dim=1)
        
        if self.use_dropout:
            x = self.dropout(x)
            
        return self.head(x)



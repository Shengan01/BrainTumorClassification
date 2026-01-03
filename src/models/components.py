import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class TransformerEncoderLayer(nn.Module):
    """Custom transformer layer that stores attention weights for visualization."""
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.attn_weights = None  # Store for visualization
    
    def forward(self, x):
        # Self-attention with weights
        attn_out, self.attn_weights = self.self_attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim=256, depth=4, heads=4, mlp_dim=512, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim=dim, heads=heads, mlp_dim=mlp_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)
        
        seq_len = x.size(1)
        # Interpolate pos embedding if needed
        if seq_len > self.pos_embedding.size(1):
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embedding[:, :seq_len, :]
        
        x = x + pos_emb
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
    def get_attention_weights(self):
        """Return attention weights from all layers."""
        weights = []
        for layer in self.layers:
            if layer.attn_weights is not None:
                weights.append(layer.attn_weights.detach())
        return weights if weights else None

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)

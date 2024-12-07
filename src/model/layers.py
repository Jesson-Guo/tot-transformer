import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.layers import Mlp


class FeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_heads=8, mlp_ratio=4):
        super(FeatureFusion, self).__init__()
        # 1x1 convolutions for channel alignment
        self.channel_align_conv1 = nn.Conv2d(in_channels=in_channels_list[0], out_channels=out_channels, kernel_size=1)
        self.channel_align_conv2 = nn.Conv2d(in_channels=in_channels_list[1], out_channels=out_channels, kernel_size=1)
        self.channel_align_conv3 = nn.Conv2d(in_channels=in_channels_list[2], out_channels=out_channels, kernel_size=1)

        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads)
        self.mlp = Mlp(out_channels, out_channels * mlp_ratio)

    def forward(self, F1, F2, F3):
        # Channel Alignment
        F1_aligned = self.channel_align_conv1(F1)
        F2_aligned = self.channel_align_conv2(F2)
        F3_aligned = self.channel_align_conv3(F3)

        # Flatten spatial dimensions and concatenate features
        F1_flat = F1_aligned.flatten(2).permute(2, 0, 1)  # Shape: (H1*W1, B, C)
        F2_flat = F2_aligned.flatten(2).permute(2, 0, 1)  # Shape: (H2*W2, B, C)
        F3_flat = F3_aligned.flatten(2).permute(2, 0, 1)  # Shape: (H3*W3, B, C)

        # Concatenate all features
        features = torch.cat([F1_flat, F2_flat, F3_flat], dim=0)  # Shape: (S, B, C)

        # Apply Attention Mechanism for Fusion
        F_mero, _ = self.attention(features, features, features)  # Shape: (S, B, C)
        F_mero = self.mlp(F_mero)

        return F_mero


class PyramidFusion(nn.Module):
    def __init__(self, embed_dims):
        super(PyramidFusion, self).__init__()
        # Convolution layers to process F1
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(),
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(),
        )
        # Convolution layers to process F2
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(),
        )

    def forward(self, F1, F2, F3):
        F1 = self.conv1(F1)
        F2 = self.conv2(F2)
        F_fused = F1 + F2 + F3

        # Reshape the fused feature map to [batch_size, H/4 * W/4, 256]
        batch_size, C, H, W = F_fused.size()
        F_fused = F_fused.permute(0, 2, 3, 1).contiguous().view(batch_size, H * W, C)

        return F_fused


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        intermediate = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

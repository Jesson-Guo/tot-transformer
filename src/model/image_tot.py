
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.backbone import Swin, SMT
from timm.layers import Mlp


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super(CrossAttention, self).__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, F_base, E_agg):
        """
        F_base: (batch_size, H*W, embed_dim)
        E_agg: (batch_size, embed_dim)
        """
        # Expand E_agg to match the spatial dimensions
        E_agg = E_agg.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # Compute multi-head attention
        attn_output, _ = self.multihead_attn(F_base, E_agg, E_agg)  # (batch_size, H*W, embed_dim)

        # Residual connection and layer normalization
        F_fused = self.layer_norm(F_base + self.dropout(attn_output))  # (batch_size, H*W, embed_dim)

        # Feed-Forward Network
        F_fused = self.layer_norm(F_fused + self.dropout(self.ffn(F_fused)))  # (batch_size, H*W, embed_dim)

        return F_fused


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


class ImageToT(nn.Module):
    def __init__(self, config, num_queries, meronyms, freeze_backbone=True):
        super(ImageToT, self).__init__()
        embed_dims = config.MODEL.EMBED_DIMS
        num_heads = config.MODEL.NUM_HEADS
        mlp_ratios = config.MODEL.MLP_RATIOS
        num_decoder_layers = config.MODEL.NUM_DECODER_LAYERS
        clip_root = config.MODEL.CLIP_ROOT
        num_base_labels = config.DATASET.NUM_CLASSES

        # Load CLIP model for text embeddings
        clip_model, _ = clip.load(clip_root, device='cpu')
        text_tokens = clip.tokenize(meronyms)  # include 'no object' meronym class
        text_weights = clip_model.encode_text(text_tokens).float()
        text_weights = F.normalize(text_weights, p=2, dim=-1)
        self.text_embeddings = nn.Embedding(len(meronyms), embed_dims[1])
        self.text_embeddings.weight.data.copy_(text_weights)

        # Backbone
        if config.MODEL.BACKBONE.NAME == "swin":
            self.backbone = Swin(config.MODEL.BACKBONE, config.DATASET.IMAGE_SIZE)
        elif config.MODEL.BACKBONE.NAME == "smt":
            self.backbone = SMT(config.MODEL.BACKBONE, config.DATASET.IMAGE_SIZE)

        # MeronymModule
        in_channels_list = [embed_dims[0] // 4, embed_dims[0] // 2, embed_dims[0]]
        self.fusion = PyramidFusion(in_channels_list)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dims[0], nhead=num_heads[0], batch_first=True),
            num_layers=num_decoder_layers
        )
        self.query_embed = nn.Embedding(num_queries, embed_dims[0])
        self.meronym_class_head = nn.Linear(embed_dims[0], len(meronyms))
        self.mlp = Mlp(embed_dims[1], embed_dims[1] * mlp_ratios[0], out_features=num_base_labels)

        # BaseModule
        self.attn = CrossAttention(embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1])
        self.base_class_head = nn.Linear(embed_dims[1], num_base_labels)

        self.text_embeddings.weight.requires_grad = False
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Backbone forward pass
        F1, F2, F3, F_base = self.backbone(x)

        # MeronymModule
        F_mero = self.fusion(F1, F2, F3)  # [batch_size, 196, 256]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(F_mero.shape[0], 1, 1)  # [batch_size, 10, 256]
        decoded_queries = self.transformer_decoder(query_embed, F_mero)
        mero_logits = self.meronym_class_head(decoded_queries)

        # Calculate base label probabilities given meronym labels
        P_mero = F.softmax(mero_logits, dim=-1)
        P_k = P_mero.sum(dim=1)  # Sum over queries to get per-meronym-class embeddings
        E_k = torch.einsum('bm,md->bmd', P_k, self.text_embeddings.weight)  # Compute per-meronym-class embeddings
        base_given_mero_logits = self.mlp(E_k)

        # Weighted Embedding Aggregation
        E_mero = torch.einsum('bqm,md->bqd', P_mero, self.text_embeddings.weight)
        E_agg = E_mero.mean(dim=1)

        # BaseModule
        F_fused = self.attn(F_base, E_agg)
        base_logits = self.base_class_head(F_fused.mean(dim=1))

        return {
            "mero": mero_logits,
            "base": base_logits,
            "base_given_mero": base_given_mero_logits
        }

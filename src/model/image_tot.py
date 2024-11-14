import torch
import torch.nn as nn
import torch.nn.functional as F
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator
from timm.models.swin_transformer import SwinTransformerStage
from timm.layers import PatchEmbed, to_ntuple
from typing import Union, List, Tuple, Optional, Callable


class SwinStage(nn.Module):
    def __init__(self, img_size, patch_size, in_chans) -> None:
        super().__init__()


class Backbone(nn.Module):
    def __init__(self, config, **kwargs,):
        super(Backbone, self).__init__()
        img_size = kwargs['img_size']
        patch_size = config.PATCH_SIZE
        in_chans = config.IN_CHANS
        prompt_width = config.PROMPT_WIDTH
        prompt_dim = config.PROMPT_DIM  # Width of prompt embeddings
        embed_dim = config.EMBED_DIM
        depths = config.DEPTHS
        num_heads = config.NUM_HEADS
        window_size = config.WINDOW_SIZE
        mlp_ratio = config.MLP_RATIO
        qkv_bias=config.QKI_BIAS
        proj_drop_rate = config.PROJ_DROP
        attn_drop_rate = config.ATTN_DROP
        drop_path_rate = config.DROP_PATH
        norm_layer = nn.LayerNorm

        self.num_stages = len(depths)

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_stages)]
        if not isinstance(window_size, (tuple, list)):
            window_size = to_ntuple(self.num_stages)(window_size)
        mlp_ratio = to_ntuple(self.num_stages)(mlp_ratio)

        # Initialize prompt embeddings for each stage
        self.prompt_embeddings = nn.ParameterList()
        for i in range(self.num_stages):
            embed_shape = (1, prompt_dim, prompt_width, prompt_width)
            prompt_embed = nn.Parameter(torch.randn(embed_shape))
            self.prompt_embeddings.append(prompt_embed)

        # Initialize projection layers to adjust channel dimensions after concatenation
        self.proj_layers = nn.ModuleList()
        for i in range(self.num_stages):
            total_in_channels = embed_dim[i] + prompt_dim
            proj_layer = nn.Conv2d(total_in_channels, embed_dim[i], kernel_size=1)
            self.proj_layers.append(proj_layer)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt='NHWC',
        )
        self.patch_grid = self.patch_embed.grid_size

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        # Initialize stages
        in_dim = embed_dim[0]
        scale = 1
        self.stages = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(self.num_stages):
            out_dim = embed_dim[i]
            stage = SwinTransformerStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_grid[0] // scale,
                    self.patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            self.stages.append(stage)
            self.norm.append(norm_layer(int(embed_dim * 2 ** i)))

            in_dim = out_dim
            if i > 0:
                scale *= 2

    def forward(self, x):
        x = self.patch_embed(x)
        features = []
        for i, stage in enumerate(self.stages):
            prompt = self.prompt_embeddings[i].expand(x.shape[0], -1, -1, -1)
            x = torch.cat((x, prompt), dim=1)  # (B, C + prompt_dim, H, W)
            x = self.proj_layers[i](x)  # (B, C, H, W)
            x = stage(x)
            features.append(self.norm[i](x))
        return features  # List of features from each stage: [F_0, F_1, F_2, F_3]

    def load_pretrained(self, pretrained_state_dict=None, pretrained=True):
        """
        Load pretrained weights for Swin Transformer stages from a pretrained Swin Transformer model.

        Args:
            pretrained_state_dict (str): Pre-trained weights.
            pretrained (bool): Whether to load pretrained weights.
        """
        if not pretrained:
            print("Pretrained weights not requested.")
            return

        # Map the pretrained weights to our model
        own_state_dict = self.state_dict()
        own_keys = list(own_state_dict.keys())

        # Remove prompt embeddings and projection layers from own_state_dict keys
        own_keys = [k for k in own_keys if not k.startswith('prompt_embeddings') and not k.startswith('proj_layers') and not k.startswith('norm')]

        # Create a mapping from pretrained keys to own keys
        mapping = {}
        for own_key in own_keys:
            # Adjust the key to match pretrained model's naming convention
            pretrained_key = own_key
            if own_key.startswith('stages.'):
                pretrained_key = own_key.replace('stages.', 'layers.')
            if pretrained_key in pretrained_state_dict:
                mapping[own_key] = pretrained_key
            else:
                print(f"Key {pretrained_key} not found in pretrained model.")

        # Load the weights
        for own_key, pretrained_key in mapping.items():
            own_state_dict[own_key] = pretrained_state_dict[pretrained_key]

        # Load the updated state_dict into the model
        self.load_state_dict(own_state_dict)
        print("Pretrained weights loaded successfully into Backbone.")


class ImageToT(nn.Module):
    def __init__(self, config):
        super(ImageToT, self).__init__()
        self.config = config
        self.backbone = Backbone(config.BACKBONE)
        self.thought_generator = ThoughtGenerator(config)
        self.state_evaluator = StateEvaluator(config)

    def forward(self, x):
        """
        x: Input image tensor
        Returns:
            preds: List of predictions from each stage
            probs: List of probability distributions from each stage
        """
        # Pass the image through the ThoughtGenerator to get features from each stage
        F_list, logits_list = self.thought_generator(x)  # List of features from each stage

        # Pass the features to the StateEvaluator to get logits
        similarities = self.state_evaluator(F_list)  # List of logits from each stage

        return logits_list, similarities

    def load_pretrained(self, backbone_path, clip_root, clip_model_name):
        """
        Load pre-trained parameters for SwinTransformerStage and CLIP's TextEncoder.
        """
        backbone_state_dict = torch.load(backbone_path, map_location='cpu')
        self.backbone.load_pretrained(backbone_state_dict)
        self.state_evaluator.load_pretrained(clip_root, clip_model_name)

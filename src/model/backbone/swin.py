import torch
import torch.nn as nn
from timm.models.swin_transformer import PatchMerging, SwinTransformerBlock
from timm.layers import PatchEmbed, to_ntuple
from timm.models.layers import to_2tuple
from typing import Union, List, Tuple, Optional, Callable


class SwinTransformerStage(nn.Module):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: Union[int, Tuple[int, int]] = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class Swin(nn.Module):
    def __init__(self, config, **kwargs,):
        super(Swin, self).__init__()
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
            embed_shape = (1, prompt_dim[i], prompt_width, prompt_width)
            prompt_embed = nn.Parameter(torch.randn(embed_shape))
            self.prompt_embeddings.append(prompt_embed)

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

            in_dim = out_dim
            if i > 0:
                scale *= 2

    def forward(self, x):
        x = self.patch_embed(x)
        features = []
        prev_prompt = None
        for i, stage in enumerate(self.stages):
            prompt = self.prompt_embeddings[i].expand(x.shape[0], -1, -1, -1)
            x = torch.cat((x, prompt), dim=1)
            x = stage(x)
            features.append(x)
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

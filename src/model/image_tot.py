import torch.nn as nn
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator


class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.num_stages = config.get('num_stages', 4)
        self.embed_width = config.get('embed_width', 1)
        self.prompt_dim = config.get('prompt_dim', 64)  # Width of prompt embeddings
        self.in_channels_list = config['in_channels_list']
        self.out_channels_list = config['out_channels_list']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_sizes = config['window_sizes']
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.norm_layer = config.get('norm_layer', nn.LayerNorm)
        
        # Initialize prompt embeddings for each stage
        self.prompt_embeddings = nn.ParameterList()
        for i in range(self.num_stages):
            embed_shape = (1, self.prompt_dim, self.embed_width, self.embed_width)
            prompt_embed = nn.Parameter(torch.randn(embed_shape))
            self.prompt_embeddings.append(prompt_embed)
        
        # Initialize projection layers to adjust channel dimensions after concatenation
        self.proj_layers = nn.ModuleList()
        for i in range(self.num_stages):
            total_in_channels = self.in_channels_list[i] + self.prompt_dim
            proj_layer = nn.Conv2d(total_in_channels, self.in_channels_list[i], kernel_size=1)
            self.proj_layers.append(proj_layer)
        
        # Initialize stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = SwinTransformerStage(
                dim=self.in_channels_list[i],
                depth=self.depths[i],
                num_heads=self.num_heads[i],
                window_size=self.window_sizes[i],
                mlp_ratio=self.mlp_ratio,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=self.norm_layer,
                downsample=nn.Conv2d(self.in_channels_list[i], self.out_channels_list[i], kernel_size=1) if self.in_channels_list[i] != self.out_channels_list[i] else None,
                use_checkpoint=False
            )
            self.stages.append(stage)

    def forward(self, x):
        features = []
        for i, stage in enumerate(self.stages):
            # Get x dimensions
            B, C, H, W = x.size()
            # Get prompt embedding
            prompt = self.prompt_embeddings[i].expand(B, -1, -1, -1)
            if self.embed_width != H or self.embed_width != W:
                prompt = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
            # Concatenate prompt embedding with x along the channel dimension
            x = torch.cat((x, prompt), dim=1)  # (B, C + prompt_dim, H, W)
            # Apply projection layer to adjust channel dimensions
            x = self.proj_layers[i](x)  # (B, C, H, W)
            # Pass through Swin Transformer Stage
            x = stage(x)
            features.append(x)
        return features  # List of features from each stage: [F_0, F_1, F_2, F_3]

    def load_pretrained(self, pretrained_state_dict=None, pretrained=True):
        """
        Load pretrained weights for Swin Transformer stages from a pretrained Swin Transformer model.

        Args:
            model_name (str): Name of the Swin Transformer model in 'timm' to load pre-trained weights from.
            pretrained (bool): Whether to load pretrained weights.
        """
        if not pretrained:
            print("Pretrained weights not requested.")
            return

        # Map the pretrained weights to our model
        own_state_dict = self.state_dict()
        own_keys = list(own_state_dict.keys())

        # Remove prompt embeddings and projection layers from own_state_dict keys
        own_keys = [k for k in own_keys if not k.startswith('prompt_embeddings') and not k.startswith('proj_layers')]

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
        self.thought_generator = ThoughtGenerator(config.THOUGHT_GENERATOR)
        self.state_evaluator = StateEvaluator(config.STATE_EVALUATOR)

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

    def load_pretrained(self):
        """
        Load pre-trained parameters for SwinTransformerStage and CLIP's TextEncoder.
        """
        backbone_state_dict = 
        clip_root = 
        clip_model_name = 
        self.backbone.load_pretrained(backbone_state_dict)
        self.state_evaluator.load_pretrained(clip_root, clip_model_name)

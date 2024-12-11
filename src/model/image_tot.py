import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.backbone import Swin, SMT
from timm.layers import Mlp
from .layers import TransformerDecoder, PyramidFusion, PositionalEncoding


class MeroBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries, num_classes, num_decoder_layers, aux_loss=False):
        super(MeroBlock, self).__init__()
        self.mero_decoder = TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(embed_dim),
            return_intermediate=aux_loss
        )
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.aux_loss = aux_loss

    def forward(self, F_mero):
        queries = self.query_embed.weight.unsqueeze(0).repeat(F_mero.shape[0], 1, 1)
        outputs = self.mero_decoder(queries, F_mero)
        outputs = self.head(outputs)
        if self.aux_loss:
            return {
                "mero": F_mero,
                "mero_logits": outputs[-1],
                "aux_outputs": [a for a in outputs[:-1]]
            }
        else:
            return {
                "mero": F_mero,
                "mero_logits": outputs[-1]
            }


class BaseBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_encoder_layers):
        super(BaseBlock, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.positional_encoding = PositionalEncoding(embed_dim)
        # self.base_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True), 
        #     num_layers=num_encoder_layers
        # )
        self.base_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True), 
            num_layers=num_encoder_layers
        )
        self.head = nn.Linear(embed_dim, num_classes)

    # def forward(self, E_mero, F_base):
    #     cls_token = self.cls_token.expand(F_base.shape[0], -1, -1)
    #     F_base = torch.cat([cls_token, E_mero, F_base], dim=1)
    #     F_base = self.positional_encoding(F_base)
    #     F_base = self.base_encoder(F_base)
    #     F_base = F_base[:, 0]
    #     # F_base = F_base.mean(dim=1)
    #     base_logits = self.head(F_base)

    def forward(self, E_mero, F_base):
        cls_token = self.cls_token.expand(F_base.shape[0], -1, -1)
        x = torch.cat([cls_token, E_mero], dim=1)
        x = self.base_decoder(x, F_base)
        x = x[:, 0]
        base_logits = self.head(x)

        return base_logits


class ImageToT(nn.Module):
    def __init__(self, config, num_queries, num_mero_classes, text_weights, freeze_backbone=True):
        super(ImageToT, self).__init__()
        embed_dims = config.MODEL.EMBED_DIMS
        num_heads = config.MODEL.NUM_HEADS
        mlp_ratios = config.MODEL.MLP_RATIOS
        num_encoder_layers = config.MODEL.NUM_ENCODER_LAYERS
        num_decoder_layers = config.MODEL.NUM_DECODER_LAYERS
        num_mero_classes = num_mero_classes
        num_base_classes = config.DATASET.NUM_CLASSES

        # Load CLIP model for text embeddings
        self.text_embeddings = nn.Embedding(num_mero_classes + 1, embed_dims[1])
        self.text_embeddings.weight.data.copy_(text_weights)

        # Backbone
        if config.MODEL.BACKBONE.NAME == "swin":
            self.backbone = Swin(config.MODEL.BACKBONE, config.DATASET.IMAGE_SIZE)
        elif config.MODEL.BACKBONE.NAME == "smt":
            self.backbone = SMT(config.MODEL.BACKBONE, config.DATASET.IMAGE_SIZE)

        self.fusion = PyramidFusion([embed_dims[0] // 4, embed_dims[0] // 2, embed_dims[0]])

        # Meronym Block
        self.mero_block = MeroBlock(
            embed_dim=embed_dims[0],
            num_heads=num_heads[0],
            num_queries=num_queries,
            num_classes=num_mero_classes + 1,
            num_decoder_layers=num_decoder_layers,
            aux_loss=config.LOSS.AUX_LOSS
        )
        self.mlp = Mlp(embed_dims[1], embed_dims[1] * mlp_ratios[0], out_features=num_base_classes)

        # BaseModule
        self.base_block = BaseBlock(embed_dims[1], num_heads[1], num_base_classes, num_encoder_layers)

        # freeze model
        self.text_embeddings.weight.requires_grad = False
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # for param in self.backbone.patch_embed4.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.block4.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.norm4.parameters():
        #     param.requires_grad = True

    def forward(self, x, E_mero):
        F1, F2, F3, F_base = self.backbone(x)
        # F1, F2, F3 = self.backbone.forward_mero(x)
        F_mero = self.fusion(F1, F2, F3)

        # Meronym Block
        mero_outputs = self.mero_block(F_mero)

        # Calculate base label probabilities given meronym labels
        P_mero = F.softmax(mero_outputs["mero_logits"], dim=-1)
        P_k = P_mero.sum(dim=1)  # Sum over queries to get per-meronym-class embeddings
        E_k = torch.einsum('bm,md->bmd', P_k, self.text_embeddings.weight)  # Compute per-meronym-class embeddings
        base_given_mero_logits = self.mlp(E_k)

        # Base Block
        # F_base = self.backbone.forward_base(F_mero)
        base_logits = self.base_block(E_mero, F_base)

        return {
            "mero": mero_outputs,
            "base": base_logits,
            "base_given_mero": base_given_mero_logits
        }

    def load_pretrained(self, model_root):
        checkpoint = torch.load(model_root, map_location='cpu')
        pretrained_state_dict = checkpoint["model"]
        self.base_block.head.load_state_dict({'weight': pretrained_state_dict['head.weight'], 'bias': pretrained_state_dict['head.bias']})

        del pretrained_state_dict['head.weight'], pretrained_state_dict['head.bias']
        msg = self.backbone.load_state_dict(pretrained_state_dict)
        return msg

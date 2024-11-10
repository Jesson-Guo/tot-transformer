import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from timm.models.swin_transformer import SwinTransformer, swin_base_patch4_window7_224
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import clip

class CoTTransformerStage(nn.Module):
    """
    Implements a single stage of the Multi-Stage Learnable CoT-Transformer.
    Produces intermediate features (F_t) and probability distributions (P_i).
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_labels: int):
        """
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            num_labels (int): Number of labels for the classification task at this granularity.
        """
        super(CoTTransformerStage, self).__init__()
        
        # Linear layer to process input features to hidden_dim
        self.feature_processor = nn.Linear(input_dim, hidden_dim)
        
        # Classification head to produce probability distributions P_i
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CoTTransformerStage.
        
        Args:
            x (torch.Tensor): Input features from Swin Transformer stage.
        
        Returns:
            F_t (torch.Tensor): Processed intermediate features.
            P_i (torch.Tensor): Probability distributions over labels.
        """
        # Process features
        F_t = self.feature_processor(x)  # Shape: (batch_size, hidden_dim)
        
        # Compute logits and probability distributions
        logits = self.classifier(F_t)    # Shape: (batch_size, num_labels)
        P_i = F.log_softmax(logits, dim=-1)  # Log probabilities for numerical stability
        
        return F_t, P_i

class StateEvaluatorStage(nn.Module):
    """
    Implements a single stage of the Multi-Stage State Evaluator.
    Evaluates the current state based on intermediate features and previous probability distributions.
    Optionally uses CLIP's Text Encoder based on the clip_mode parameter.
    """
    def __init__(self, 
                 hidden_dim: int, 
                 num_labels: int,
                 total_labels: int,
                 label_descriptions: Optional[List[str]] = None,
                 clip_mode: Optional[str] = None,  # None, 'label_descriptions', 'process_p_prev'
                 eval_hidden_dim: int = 256,
                 device: str = 'cpu'):
        """
        Args:
            hidden_dim (int): Dimension of the intermediate features F_t.
            num_labels (int): Number of labels at current granularity level.
            total_labels (int): Total number of labels from all previous stages.
            label_descriptions (List[str], optional): List of label descriptions for Approach 1.
            clip_mode (str, optional): Mode to use CLIP. Options: None, 'label_descriptions', 'process_p_prev'.
            eval_hidden_dim (int): Dimension of the hidden layer in the evaluator.
            device (str): Device to load CLIP model ('cpu' or 'cuda').
        """
        super(StateEvaluatorStage, self).__init__()
        self.clip_mode = clip_mode
        self.device = device
        
        # Fully connected layer to process F_t
        self.fc_F = nn.Linear(hidden_dim, eval_hidden_dim)
        
        # Depending on the mode, initialize different components
        if self.clip_mode == 'label_descriptions':
            assert label_descriptions is not None, "label_descriptions must be provided when clip_mode='label_descriptions'"
            # Load CLIP model
            self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
            self.clip_model.eval()  # Set to evaluation mode
            # Encode label descriptions
            with torch.no_grad():
                text_tokens = clip.tokenize(label_descriptions).to(device)  # Shape: (num_labels, token_length)
                self.text_embeddings = self.clip_model.encode_text(text_tokens)  # Shape: (num_labels, clip_dim)
            # Project CLIP embeddings to desired dimension
            clip_dim = self.text_embeddings.shape[1]
            self.fc_clip = nn.Linear(clip_dim, eval_hidden_dim)
        elif self.clip_mode == 'process_p_prev':
            # Load CLIP model
            self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
            self.clip_model.eval()  # Set to evaluation mode
            # Assume label descriptions are the class names up to total_labels
            # These should be provided externally or derived from the dataset
            # For simplicity, we'll assume label_descriptions is a list of all labels up to current stage
            # and passed via label_descriptions parameter
            assert label_descriptions is not None, "label_descriptions must be provided when clip_mode='process_p_prev'"
            with torch.no_grad():
                text_tokens = clip.tokenize(label_descriptions).to(device)  # Shape: (total_labels, token_length)
                self.text_embeddings = self.clip_model.encode_text(text_tokens)  # Shape: (total_labels, clip_dim)
            # Project CLIP embeddings to desired dimension
            clip_dim = self.text_embeddings.shape[1]
            self.fc_clip = nn.Linear(clip_dim, eval_hidden_dim)
        elif self.clip_mode is None:
            # Linear layer to process P_prev without CLIP
            self.fc_P = nn.Linear(total_labels, eval_hidden_dim)
        else:
            raise ValueError("Invalid clip_mode. Options are: None, 'label_descriptions', 'process_p_prev'")
        
        # Determine combined input dimension
        if self.clip_mode == 'label_descriptions':
            combined_input_dim = eval_hidden_dim * 2  # F_processed + Text_processed
        elif self.clip_mode == 'process_p_prev':
            combined_input_dim = eval_hidden_dim * 2  # F_processed + Text_processed
        else:
            combined_input_dim = eval_hidden_dim + eval_hidden_dim  # F_processed + P_processed
        
        # Combined layers
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_dim, eval_hidden_dim),
            nn.ReLU(),
            nn.Linear(eval_hidden_dim, num_labels)
        )
        
    def forward(self, 
                F_t: torch.Tensor, 
                P_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StateEvaluatorStage.

        Args:
            F_t (torch.Tensor): Intermediate features from CoTTransformerStage.
            P_prev (torch.Tensor): Concatenated log probability distributions from all previous stages.

        Returns:
            P_t (torch.Tensor): Updated log probability distribution at current granularity level.
        """
        F_processed = self.fc_F(F_t)  # Shape: (batch_size, eval_hidden_dim)
        
        if self.clip_mode == 'label_descriptions':
            # Process label descriptions via CLIP
            # self.text_embeddings: (num_labels, clip_dim)
            text_processed = self.fc_clip(self.text_embeddings)  # Shape: (num_labels, eval_hidden_dim)
            # Compute similarity between F_processed and text_processed
            # Expand F_processed to (batch_size, 1, eval_hidden_dim)
            F_expanded = F_processed.unsqueeze(1)  # Shape: (batch_size, 1, eval_hidden_dim)
            # text_processed: (1, num_labels, eval_hidden_dim)
            text_expanded = text_processed.unsqueeze(0)  # Shape: (1, num_labels, eval_hidden_dim)
            # Compute dot product
            logits = torch.bmm(F_expanded, text_expanded.transpose(1, 2)).squeeze(1)  # Shape: (batch_size, num_labels)
            P_t = F.log_softmax(logits, dim=-1)
        elif self.clip_mode == 'process_p_prev':
            # Convert log probabilities to probabilities
            P_prev_probs = P_prev.exp()  # Shape: (batch_size, total_labels)
            # Compute weighted sum of text embeddings
            # self.text_embeddings: (total_labels, clip_dim)
            weighted_text_embeddings = torch.matmul(P_prev_probs, self.text_embeddings)  # Shape: (batch_size, clip_dim)
            # Project text embeddings
            text_processed = self.fc_clip(weighted_text_embeddings)  # Shape: (batch_size, eval_hidden_dim)
            # Combine with F_processed
            combined = torch.cat([F_processed, text_processed], dim=-1)  # Shape: (batch_size, eval_hidden_dim * 2)
            logits = self.combined_fc(combined)  # Shape: (batch_size, num_labels)
            P_t = F.log_softmax(logits, dim=-1)
        else:
            # Process P_prev without CLIP
            P_processed = self.fc_P(P_prev)  # Shape: (batch_size, eval_hidden_dim)
            # Combine with F_processed
            combined = torch.cat([F_processed, P_processed], dim=-1)  # Shape: (batch_size, eval_hidden_dim * 2)
            logits = self.combined_fc(combined)  # Shape: (batch_size, num_labels)
            P_t = F.log_softmax(logits, dim=-1)
        
        return P_t

class MultiStageCoT(nn.Module):
    """
    Implements the entire Multi-Stage Learnable CoT-Transformer model, integrating multiple CoTTransformerStage and StateEvaluatorStage.
    Supports optional integration of CLIP's Text Encoder through different modes.
    """
    def __init__(self, 
                 num_labels_list: List[int],
                 label_descriptions_list: Optional[List[List[str]]] = None,  # List per stage
                 clip_mode_list: Optional[List[Optional[str]]] = None,       # List per stage
                 pretrained: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            num_labels_list (List[int]): List of label counts for each granularity level (4 elements).
            label_descriptions_list (List[List[str]], optional): List containing lists of label descriptions for each stage.
                Required if any stage uses 'label_descriptions' or 'process_p_prev' mode.
            clip_mode_list (List[Optional[str]], optional): List containing clip_mode for each stage.
                Each element should be one of: None, 'label_descriptions', 'process_p_prev'.
                Required if label_descriptions_list is provided.
            pretrained (bool): Whether to use pretrained weights for Swin Transformer.
            device (str): Device to load CLIP model ('cpu' or 'cuda').
        """
        super(MultiStageCoT, self).__init__()
        self.num_stages = 4  # As per Swin Transformer stages
        assert len(num_labels_list) == self.num_stages, "num_labels_list must have 4 elements."
        
        if label_descriptions_list is not None:
            assert len(label_descriptions_list) == self.num_stages, "label_descriptions_list must have 4 elements."
            assert clip_mode_list is not None, "clip_mode_list must be provided if label_descriptions_list is provided."
            assert len(clip_mode_list) == self.num_stages, "clip_mode_list must have 4 elements."
            # Validate clip_mode values
            valid_modes = [None, 'label_descriptions', 'process_p_prev']
            for mode in clip_mode_list:
                assert mode in valid_modes, f"Invalid clip_mode: {mode}. Valid options are: {valid_modes}"
        else:
            clip_mode_list = [None] * self.num_stages
        
        self.device = device
        
        # Initialize Swin Transformer backbone
        self.swin = swin_base_patch4_window7_224(pretrained=pretrained)
        
        # Extract feature dimensions at each stage
        # Swin Transformer stages output feature dimensions as per embed_dim * 2^i
        # For standard Swin-T, embed_dim=96, layers=[2, 2, 6, 2], so stage_dims = [96, 192, 384, 768]
        self.stage_dims = [96 * (2 ** i) for i in range(self.num_stages)]  # Adjust based on actual Swin config
        
        # Initialize CoTTransformerStages and StateEvaluatorStages
        self.cot_stages = nn.ModuleList()
        self.state_evaluators = nn.ModuleList()
        
        # Keep track of total labels up to the current stage for 'process_p_prev' mode
        total_labels_so_far = 0
        for t in range(self.num_stages):
            # Calculate input_dim by summing dims from Swin stages t to end
            input_dims = sum(self.stage_dims[t:])
            # CoTTransformerStage
            cot_stage = CoTTransformerStage(
                input_dim=input_dims,
                hidden_dim=256,
                num_labels=num_labels_list[t]
            )
            self.cot_stages.append(cot_stage)
            
            total_labels_so_far += num_labels_list[t]
            # StateEvaluatorStage
            clip_mode = clip_mode_list[t]
            if clip_mode == 'process_p_prev':
                # For 'process_p_prev', label_descriptions should include all labels up to current stage
                # Assuming label_descriptions_list[t] contains descriptions for all labels up to current stage
                # This requires that label_descriptions_list[t] has cumulative label descriptions
                label_descriptions = label_descriptions_list[t]
            elif clip_mode == 'label_descriptions':
                # For 'label_descriptions', label_descriptions_list[t] contains descriptions for current stage's labels
                label_descriptions = label_descriptions_list[t]
            else:
                label_descriptions = None
            state_evaluator = StateEvaluatorStage(
                hidden_dim=256,
                num_labels=num_labels_list[t],
                total_labels=total_labels_so_far,
                label_descriptions=label_descriptions,
                clip_mode=clip_mode,
                eval_hidden_dim=256,
                device=device
            )
            self.state_evaluators.append(state_evaluator)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the MultiStageCoT.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).

        Returns:
            P_all (List[torch.Tensor]): List of log probability distributions from all stages.
        """
        # Swin Transformer forward pass to extract features at each stage
        features = self.swin.forward_features(x)  # List of features per stage
        
        P_all = []      # List to store log probability distributions
        P_concat = []   # List to store concatenated log probabilities
        
        for t in range(self.num_stages):
            # Aggregate features from Swin stages t to end
            agg_features = []
            for s in range(t, self.num_stages):
                F_swin = features[s]  # Shape: (batch_size, H_s * W_s, C_s)
                # Global average pooling
                F_swin = F_swin.mean(dim=1)  # Shape: (batch_size, C_s)
                agg_features.append(F_swin)
            # Concatenate aggregated features
            F_input = torch.cat(agg_features, dim=-1)  # Shape: (batch_size, sum(C_s))
            
            # CoTTransformerStage
            F_t, P_i = self.cot_stages[t](F_input)
            P_all.append(P_i)
            P_concat.append(P_i)
            # Concatenate all previous P_i
            P_prev_concat = torch.cat(P_concat, dim=-1)  # Shape: (batch_size, sum(num_labels up to t))
            
            # StateEvaluatorStage
            P_t = self.state_evaluators[t](F_t, P_prev_concat)
            # Update the current P_i with evaluated P_t
            P_all[-1] = P_t
            P_concat[-1] = P_t
        
        return P_all  # List of log probability distributions

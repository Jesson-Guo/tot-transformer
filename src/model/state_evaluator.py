import torch
import torch.nn as nn
import clip


class StateAttention(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super(StateAttention, self).__init__()
        self.W_Q = nn.Linear(input_dim, proj_dim)
        self.W_K = nn.Linear(input_dim, proj_dim)
        self.W_V = nn.Linear(input_dim, proj_dim)
        self.scale = torch.sqrt(torch.tensor(proj_dim, dtype=torch.float32))

    def forward(self, Q, K_list, V_list):
        """
        Q: Tensor of shape (B, input_dim)
        K_list: List of tensors [(B, input_dim), ...]
        V_list: List of tensors [(B, input_dim), ...]
        """
        # Project Q
        Q_proj = self.W_Q(Q)  # (B, proj_dim)

        # Project K and V
        K_proj_list = [self.W_K(K) for K in K_list]
        V_proj_list = [self.W_V(V) for V in V_list]

        # Stack K and V
        K_proj = torch.stack(K_proj_list, dim=1)  # (B, t-1, proj_dim)
        V_proj = torch.stack(V_proj_list, dim=1)  # (B, t-1, proj_dim)

        # Compute attention scores
        attn_scores = torch.bmm(Q_proj.unsqueeze(1), K_proj.transpose(1, 2)) / self.scale  # (B, 1, t-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, 1, t-1)

        # Compute context vector
        context = torch.bmm(attn_weights, V_proj).squeeze(1)  # (B, proj_dim)

        # Combine Q and context
        H = Q_proj + context  # (B, proj_dim)
        return H


class StateEvaluator(nn.Module):
    def __init__(self, config, **kwargs,):
        super(StateEvaluator, self).__init__()
        self.num_stages = 4
        embed_dim = config.EMBED_DIM
        self.use_clip = config.USE_CLIP
        self.label_descriptions_list = kwargs['label_descriptions_list']
        self.eval_hidden_dim = config.HIDDEN_DIM
        self.proj_dim = config.PROJ_DIM

        # Initialize lists to hold per-stage components
        self.image_projections = nn.ModuleList()
        self.state_attentions = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        if self.use_clip:
            # Placeholders for CLIP components; will be initialized in load_pretrained
            self.clip_model = None
            self.text_embeddings_list = []
            self.text_projections = nn.ModuleList()
        else:
            self.clip_model = None
            self.text_embeddings_list = None
            self.text_projections = None

        # Initialize per-stage components
        for i in range(self.num_stages):
            # Projection layer for image features
            img_proj = nn.Linear(embed_dim[i], self.eval_hidden_dim[i])
            self.image_projections.append(img_proj)

            # State attention mechanism
            state_attn = StateAttention(self.eval_hidden_dim[i], self.proj_dim[i])
            self.state_attentions.append(state_attn)

            # Output layer
            output_layer = nn.Linear(self.proj_dim[i], self.eval_hidden_dim[i])
            self.output_layers.append(output_layer)

    def _encode_label_descriptions(self, label_descriptions):
        with torch.no_grad():
            text_tokens = clip.tokenize(label_descriptions)
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def forward(self, F_list):
        """
        Args:
            F_list: List of feature tensors [(B, C, H, W), ...] from ThoughtGenerator

        Returns:
            similarities: List of similarity scores for each stage
        """
        similarities = []
        F_prev_proj_list = []

        for t in range(self.num_stages):
            F_t = F_list[t]

            # Global average pooling
            F_t_pooled = F_t.mean(dim=[2, 3])  # Shape: (B, C)
            # Project the pooled features
            F_t_proj = self.image_projections[t](F_t_pooled)  # Shape: (B, eval_hidden_dim)

            # Prepare previous projected features
            if t == 0:
                F_prev_proj_list = []

            # Apply state attention
            H = self.state_attentions[t](F_t_proj, F_prev_proj_list, F_prev_proj_list)  # Shape: (B, proj_dim)

            # Compute similarity scores
            if self.use_clip:
                if self.clip_model is None or not self.text_embeddings_list:
                    raise ValueError("CLIP model and text embeddings must be loaded using load_pretrained before calling forward.")

                text_embeddings_proj = self.text_projections[t](self.text_embeddings_list[t])  # Shape: (N, proj_dim)
                text_embeddings_proj = text_embeddings_proj / text_embeddings_proj.norm(dim=-1, keepdim=True)
                H_normalized = H / H.norm(dim=-1, keepdim=True)
                similarity_scores = H_normalized @ text_embeddings_proj.T  # Shape: (B, N)
            else:
                similarity_scores = self.output_layers[t](H)  # Shape: (B, eval_hidden_dim)

            similarities.append(similarity_scores)

            # Update previous projected features
            F_prev_proj_list.append(F_t_proj)

        return similarities

    def load_pretrained(self, clip_root="", clip_model_name="ViT-B/32", device='cpu'):
        """
        Load pretrained parameters into the StateEvaluator, specifically initializing and loading the CLIP model.

        Args:
            clip_model_name (str): Name of the CLIP model to load (e.g., "ViT-B/32").
            device (str): Device to load the CLIP model onto ('cpu' or 'cuda').
        """
        if not self.use_clip:
            print("CLIP is not enabled in the configuration; cannot load CLIP model.")
            return

        # Load the CLIP model
        self.clip_model, _ = clip.load(clip_model_name, device=device, download_root=clip_root)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Initialize text embeddings and projections for each stage
        self.text_embeddings_list = []
        self.text_projections = nn.ModuleList()

        for i in range(self.num_stages):
            label_descriptions = self.label_descriptions_list[i]
            if label_descriptions is None:
                raise ValueError(f"Label descriptions for stage {i} must be provided when using CLIP.")

            # Encode label descriptions for this stage
            text_embeddings = self._encode_label_descriptions(label_descriptions)
            self.text_embeddings_list.append(text_embeddings.to(device))

            # Projection layer for text embeddings
            text_proj = nn.Linear(text_embeddings.size(1), self.proj_dim[i]).to(device)
            self.text_projections.append(text_proj)

        print("CLIP model and text embeddings loaded successfully into StateEvaluator.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinForImageClassification, CLIPModel, CLIPTokenizer
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

class StateEvaluatorStage(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', pretrained=True):
        super(StateEvaluatorStage, self).__init__()
        # Initialize CLIP Model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name) if pretrained else None
        self.clip_model.eval()  # Freeze CLIP parameters

        # Projection layer to map image features to CLIP's image embedding space
        # Assuming image_features have the same dimension as CLIP's image embeddings
        self.image_proj = nn.Linear(768, self.clip_model.config.hidden_size)
    
    def forward(self, image_features, text_input_ids, attention_mask):
        """
        image_features: Tensor [batch_size, hidden_size] (from CoTTransformerStage)
        text_input_ids: Tensor [batch_size, seq_length]
        attention_mask: Tensor [batch_size, seq_length]
        Returns:
            similarity: Tensor [batch_size, batch_size]
        """
        with torch.no_grad():
            # Project image features
            projected_image_features = self.image_proj(image_features)  # [batch_size, clip_hidden_size]
            projected_image_features = F.normalize(projected_image_features, p=2, dim=-1)
    
            # Encode text sequences
            text_outputs = self.clip_model.get_text_features(
                input_ids=text_input_ids,
                attention_mask=attention_mask
            )  # [batch_size, clip_hidden_size]
            text_embeddings = F.normalize(text_outputs, p=2, dim=-1)
    
            # Compute cosine similarity matrix
            similarity = torch.matmul(projected_image_features, text_embeddings.t())  # [batch_size, batch_size]
    
        return similarity

class CoTTransformerStage(nn.Module):
    def __init__(self, num_classes, prompt_dim, pretrained=True):
        super(CoTTransformerStage, self).__init__()
        # Initialize Swin Transformer Backbone
        self.backbone = SwinForImageClassification.from_pretrained(
            'microsoft/swin-base-patch4-window7-224' if pretrained else None,
            num_labels=0  # We'll handle classification separately
        )
        
        # Learnable Prompt
        self.prompt = nn.Parameter(torch.randn(1, prompt_dim))  # Shape: [1, prompt_dim]
        
        # Classification Head
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        x: Input images tensor
        Returns:
            logits: [batch_size, num_classes]
            features: [batch_size, hidden_size]
        """
        features = self.backbone.forward_features(x)  # Shape: [batch_size, hidden_size]
        # Integrate prompt
        prompt_expanded = self.prompt.expand(x.size(0), -1)  # Shape: [batch_size, prompt_dim]
        combined_features = features + prompt_expanded  # Simple addition; can be more complex
        logits = self.classifier(combined_features)  # Shape: [batch_size, num_classes]
        return logits, combined_features

class MultiStageCoT(nn.Module):
    def __init__(self, num_stages=4, num_classes_per_stage=[100, 200, 300, 400], prompt_dim=768, label_names_per_stage=None, clip_model_name='openai/clip-vit-base-patch32', pretrained=True):
        super(MultiStageCoT, self).__init__()
        self.num_stages = num_stages
        self.num_classes_per_stage = num_classes_per_stage
        self.label_names_per_stage = label_names_per_stage  # List of list: labels per stage
    
        # Initialize per-stage CoT Transformers and State Evaluators
        self.cot_transformers = nn.ModuleList([
            CoTTransformerStage(
                num_classes=num_classes_per_stage[t],
                prompt_dim=prompt_dim,
                pretrained=pretrained
            ) for t in range(num_stages)
        ])
        
        self.state_evaluators = nn.ModuleList([
            StateEvaluatorStage(
                clip_model_name=clip_model_name,
                pretrained=pretrained
            ) for _ in range(num_stages)
        ])
        
        # Initialize CLIP Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    
    def forward(self, x):
        """
        x: Input images tensor
        Returns:
            logits: List of logits per stage
            features: List of features per stage
            similarities: List of similarity scores per stage
        """
        logits = []
        features = []
        similarities = []
    
        for t in range(self.num_stages):
            # CoT Transformer Stage
            logit, feature = self.cot_transformers[t](x)
            logits.append(logit)
            features.append(feature)
    
            # Get probability distribution
            P_t = F.softmax(logit, dim=1)  # [batch_size, num_classes_t]
            # Get top-1 prediction
            _, preds = torch.max(P_t, dim=1)  # [batch_size]
            # Map to label names
            label_names = [self.label_names_per_stage[t][pred] for pred in preds.tolist()]
            # Create text sequences up to current stage
            # Assuming hierarchical, concatenate labels up to current stage
            text_sequences = []
            for labels in label_names:
                # For each sample, create a space-separated string of labels up to stage t
                # Example: "Animal Mammal Dog"
                seq = ' '.join(labels[:t+1])  # Adjust based on your label hierarchy
                text_sequences.append(seq)
    
            # Tokenize text sequences
            encoding = self.tokenizer(
                text_sequences,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            text_input_ids = encoding['input_ids'].to(x.device)
            attention_mask = encoding['attention_mask'].to(x.device)
    
            # State Evaluator Stage
            similarity = self.state_evaluators[t](feature, text_input_ids, attention_mask)  # [batch_size, batch_size]
            similarities.append(similarity)
    
        return logits, features, similarities

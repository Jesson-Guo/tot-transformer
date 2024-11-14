import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
    
    def forward(self, logits, soft_targets):
        """
        logits: [batch_size, num_classes]
        soft_targets: [batch_size, num_classes]
        """
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
        return loss


class ClassificationLoss(nn.Module):
    def __init__(self, num_stages, alpha, use_soft_labels=False):
        super(ClassificationLoss, self).__init__()
        self.num_stages = num_stages
        self.alpha = alpha  # List of alpha values for each stage
        self.use_soft_labels = use_soft_labels
        self.cls_loss_fn = SoftTargetCrossEntropy() if use_soft_labels else nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        logits: List of logits per stage [logit_1, logit_2, ..., logit_T]
        labels: List of ground truth labels per stage [y_1*, y_2*, ..., y_T*]
        Returns:
            L_cls: Total classification loss
        """
        L_cls = 0.0
        for t in range(self.num_stages):
            if self.use_soft_labels:
                L_cls += self.alpha[t] * self.cls_loss_fn(logits[t], labels[t])
            else:
                L_cls += self.alpha[t] * F.cross_entropy(logits[t], labels[t])

        return L_cls


class CoherenceLoss(nn.Module):
    def __init__(self, num_stages, H_matrices, beta, use_soft_labels=False):
        super(CoherenceLoss, self).__init__()
        self.num_stages = num_stages
        self.beta = beta  # List of beta values for each stage
        self.use_soft_labels = use_soft_labels
        self.H_matrices = H_matrices

    def forward(self, logits, labels):
        """
        logits: List of logits per stage [logit_1, logit_2, ..., logit_T]
        labels: List of ground truth labels per stage [y_1*, y_2*, ..., y_T*]
        Returns:
            L_coh: Total coherence loss
        """
        L_coh = 0.0
        epsilon = 1e-8  # Small value to avoid log(0)

        for t in range(1, self.num_stages):
            P_t = F.softmax(logits[t], dim=1)  # [batch_size, num_classes_t]
            P_prev = labels[t-1] if self.use_soft_labels else F.softmax(logits[t-1], dim=1)
            H = self.H_matrices[t-1].to(P_t.device)  # Move H to the device

            # Compute adjusted target distribution P_t^{adj}
            P_adj = torch.matmul(P_prev, H)  # [batch_size, num_classes_t]
    
            # Clamp P_adj to avoid log(0)
            P_adj = torch.clamp(P_adj, min=epsilon)

            # Compute KL Divergence
            L_coh += self.beta[t] * F.kl_div(P_t.log(), P_adj, reduction='batchmean')

        return L_coh


class EvaluatorLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(EvaluatorLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, positive_indices):
        """
        features: (B, D) representations from StateEvaluatorStage
        positive_indices: indices indicating positive pairs
        """
        features = features / features.norm(dim=-1, keepdim=True)
        similarity_matrix = features @ features.T  # (B, B)

        # Create labels
        labels = torch.arange(features.size(0)).to(features.device)
        labels = (labels.unsqueeze(0) == positive_indices.unsqueeze(1)).float()

        # Mask to remove self-similarity
        mask = torch.eye(features.size(0)).to(features.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # Apply temperature
        logits = similarity_matrix / self.temperature

        # Evaluator loss
        loss = -torch.sum(labels * torch.log_softmax(logits, dim=-1), dim=-1).mean()
        return loss


class ToTLoss(nn.Module):
    def __init__(self, num_stages, H_matrices, alpha, beta, gamma, lambda_eval, use_soft_labels=False):
        super(ToTLoss, self).__init__()
        self.classification_loss = ClassificationLoss(num_stages, alpha, use_soft_labels)
        self.coherence_loss = CoherenceLoss(num_stages, H_matrices, beta, use_soft_labels)
        self.evaluator_loss = EvaluatorLoss(gamma)
        self.lambda_eval = lambda_eval

    def forward(self, logits, labels, similarities):
        """
        logits: List of logits per stage
        labels: List of ground truth labels per stage
        similarities: List of similarity scores per stage
        Returns:
            L_total, L_cls, L_coh, L_evl
        """
        L_cls = self.classification_loss(logits, labels)
        L_coh = self.coherence_loss(logits, labels)
        L_evl = self.evaluator_loss(similarities)
        L_total = L_cls + L_coh + self.lambda_eval * L_evl
        return L_total, L_cls, L_coh, L_evl


class SoftToTLoss(ToTLoss):
    def __init__(self, num_stages, H_matrices, alpha, beta, gamma, lambda_eval):
        super(SoftToTLoss, self).__init__(
            num_stages=num_stages,
            H_matrices=H_matrices,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_eval=lambda_eval,
            use_soft_labels=True
        )

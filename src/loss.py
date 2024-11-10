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

class MultiStageLoss(nn.Module):
    def __init__(self, num_stages, mg_graph, alpha, beta, gamma, lambda_eval, use_soft_labels=False):
        super(MultiStageLoss, self).__init__()
        self.num_stages = num_stages
        self.mg_graph = mg_graph  # Instance of MultiGranGraph
        self.alpha = alpha  # List of alpha_i
        self.beta = beta    # List of beta_t for coherence
        self.gamma = gamma  # List of gamma_t for evaluator
        self.lambda_eval = lambda_eval
        self.use_soft_labels = use_soft_labels  # Flag to determine loss type
        
        # Initialize appropriate classification loss
        if self.use_soft_labels:
            self.cls_loss_fn = SoftTargetCrossEntropy()
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()
        
        # Precompute Hierarchical Relationship Matrices H for each stage
        self.H_matrices = []
        for t in range(1, num_stages):
            H = self.mg_graph.get_H_matrix(t)  # H matrix for stage t
            self.H_matrices.append(H)
    
    def forward(self, logits, labels, similarities):
        """
        logits: List of logits per stage [logit_1, logit_2, ..., logit_T]
        labels: List of ground truth labels per stage [y_1*, y_2*, ..., y_T*]
                Each y_t* is either a tensor of class indices or soft labels
        similarities: List of similarity scores per stage [S_1, S_2, ..., S_T]
        Returns:
            L_cls, L_coh, L_eval
        """
        # Classification Loss
        L_cls = 0.0
        for t in range(self.num_stages):
            if self.use_soft_labels:
                L_cls += self.alpha[t] * self.cls_loss_fn(logits[t], labels[t])
            else:
                L_cls += self.alpha[t] * F.cross_entropy(logits[t], labels[t])
    
        # Coherence Loss
        L_coh = 0.0
        for t in range(1, self.num_stages):
            P_t = F.softmax(logits[t], dim=1)  # [batch_size, num_classes_t]
            if self.use_soft_labels:
                P_prev = labels[t-1]  # Assuming labels[t-1] is a soft distribution
            else:
                P_prev = F.softmax(logits[t-1], dim=1)  # [batch_size, num_classes_{t-1}]
    
            H = self.H_matrices[t-1].to(P_t.device)  # [N_{t-1}, N_t}]
    
            # Compute adjusted target distribution P_t^{adj}
            P_adj = torch.matmul(P_prev, H)  # [batch_size, num_classes_t]
    
            # Clamp P_adj to avoid log(0)
            epsilon = 1e-8
            P_adj = torch.clamp(P_adj, min=epsilon)
    
            # Compute KL Divergence
            L_coh += self.beta[t] * F.kl_div(P_t.log(), P_adj, reduction='batchmean')
    
        # Evaluator Contrastive Loss
        L_eval = 0.0
        for t in range(self.num_stages):
            S_t = similarities[t]  # [batch_size, batch_size]
            # InfoNCE Loss
            # The diagonal of S_t is the similarity between correct pairs
            # Off-diagonal are similarities between incorrect pairs
            # Thus, targets are [0, 1, 2, ..., batch_size-1]
            batch_size = S_t.size(0)
            temperature = 0.07  # Can be a hyperparameter
    
            # Scale the similarities by temperature
            S_t = S_t / temperature
    
            # Targets for InfoNCE are diagonal indices
            targets = torch.arange(batch_size).to(S_t.device)
    
            # Compute cross entropy loss
            L_eval += self.gamma[t] * F.cross_entropy(S_t, targets)
    
        return L_cls, L_coh, L_eval

class SoftMultiStageLoss(MultiStageLoss):
    def __init__(self, num_stages, mg_graph, alpha, beta, gamma, lambda_eval):
        super(SoftMultiStageLoss, self).__init__(
            num_stages=num_stages,
            mg_graph=mg_graph,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_eval=lambda_eval,
            use_soft_labels=True
        )

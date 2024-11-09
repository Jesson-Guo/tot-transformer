import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiStageLoss(nn.Module):
    def __init__(self, num_stages, mg_graph, alpha, beta, gamma, lambda_eval):
        super(MultiStageLoss, self).__init__()
        self.num_stages = num_stages
        self.mg_graph = mg_graph  # Instance of MultiGranGraph
        self.alpha = alpha  # List of alpha_i
        self.beta = beta    # List of beta_t for coherence
        self.gamma = gamma  # List of gamma_t for evaluator
        self.lambda_eval = lambda_eval

        # Precompute Hierarchical Relationship Matrices H for each stage
        self.H_matrices = []
        for t in range(1, num_stages):
            H = self.mg_graph.get_H_matrix(t)  # Assuming method to get H for stage t
            self.H_matrices.append(H)

    def forward(self, P_t, labels, similarities):
        """
        P_t: List of probability distributions per stage [P_1, P_2, ..., P_T]
        labels: List of ground truth labels per stage [y_1*, y_2*, ..., y_T*]
        similarities: List of similarity scores from State Evaluator per stage [S_1, S_2, ..., S_T]
        """
        L_cls = 0.0
        for t in range(self.num_stages):
            L_cls += self.alpha[t] * F.cross_entropy(P_t[t], labels[t])

        # Coherence Loss
        L_coh = 0.0
        for t in range(1, self.num_stages):
            P_prev = P_t[t-1]  # P_{t-1}
            P_current = P_t[t]  # P_t

            H = self.H_matrices[t-1].to(P_prev.device)  # H_{t}

            # Compute adjusted target distribution P_t^{adj}
            P_adj = torch.matmul(P_prev, H)  # Shape: [batch_size, N_t]

            # To avoid log(0), add a small epsilon
            epsilon = 1e-8
            P_adj = torch.clamp(P_adj, min=epsilon)

            # Compute KL Divergence
            L_coh += self.beta[t] * F.kl_div(P_current.log(), P_adj, reduction='batchmean')

        # Evaluator Contrastive Loss
        L_eval = 0.0
        for t in range(self.num_stages):
            S_t = similarities[t]  # Similarity scores for stage t
            # InfoNCE Loss
            # Assuming S_t has shape [batch_size, batch_size] where diagonal is positive
            # and off-diagonal are negatives
            batch_size = S_t.size(0)
            temperature = 0.07  # Can be a config parameter

            # Labels for contrastive loss
            labels_contrast = torch.arange(batch_size).to(S_t.device)

            # Compute InfoNCE Loss
            L_eval += self.gamma[t] * F.cross_entropy(S_t / temperature, labels_contrast)

        return L_cls, L_coh, L_eval

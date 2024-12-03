import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, pred_logits, targets):
        """
        Performs the matching between targets and proposals.

        Parameters:
            pred_logits: Tensor of shape [batch_size, num_queries, num_classes + 1]
                (Assuming +1 for the "no-object" class if applicable)
            targets: List of targets (len(targets) = batch_size), where each target is 
                Tensor of shape [num_target_meronyms] containing the meronym class labels

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_meronyms)
        """
        bs, num_queries = pred_logits.shape[:2]

        # Flatten to compute the cost matrices in a batch
        out_prob = pred_logits.flatten(0, 1).softmax(-1)

        # Concat the target labels
        sizes = [len(tgt) for tgt in targets]
        tgt_ids = torch.cat(targets)

        # Compute the classification cost
        cost = -out_prob[:, tgt_ids]
        cost = cost.view(bs, num_queries, -1).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianLoss(nn.Module):
    """
    This class computes the Hungarian Loss for the meronym labels.

    The process involves:
        1) Computing the Hungarian matching between ground truth meronym labels and model predictions.
        2) Computing the cross-entropy loss for the matched pairs.
    """
    def __init__(self, num_classes, matcher, eos_coef=0.1):
        """
        Initializes the HungarianLoss.

        Parameters:
            num_classes: Number of meronym classes (excluding the "no-object" class).
            matcher: Instance of HungarianMatcher configured for classification only.
            eos_coef: Weight for the "no-object" class in the classification loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        # Define the weight for each class in the loss function, with the "no-object" class having a separate weight.
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef  # The last class is assumed to be "no-object"
        self.register_buffer('empty_weight', empty_weight)

    def loss_meronym_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(outputs.shape[:2], self.num_classes, dtype=torch.int64, device=outputs.device)
        target_classes[idx] = target_classes_o

        loss = F.cross_entropy(outputs.transpose(1, 2), target_classes, self.empty_weight)
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        loss = self.loss_meronym_labels(outputs, targets, indices)
        return loss


class MeronymLoss(nn.Module):
    """
    This class computes the loss for the meronym labels.
    """
    def __init__(self, num_classes, matcher, lambda_mero):
        """
        Initializes the MeronymLoss.

        Parameters:
            num_classes: Number of meronym classes (excluding the "no-object" class).
            matcher: Instance of HungarianMatcher configured for classification only.
            lambda_mero: Weighting factor for the meronym label loss.
        """
        super().__init__()
        self.hungarian_loss = HungarianLoss(num_classes, matcher)
        self.lambda_mero = lambda_mero

    def forward(self, mero_logits, mero_labels):
        loss = self.hungarian_loss(mero_logits, mero_labels)
        loss = loss * self.lambda_mero
        return loss


class BaseLoss(nn.Module):
    """
    This class computes the loss for the base labels.

    It computes the cross-entropy loss between the predicted base labels and the ground truth base labels.
    """
    def __init__(self, lambda_base=1.0):
        """
        Initializes the BaseLoss.

        Parameters:
            lambda_base: Weighting factor for the base label loss.
        """
        super().__init__()
        self.lambda_base = lambda_base

    def forward(self, base_logits, base_labels):
        loss_ce = F.cross_entropy(base_logits, base_labels)
        loss = loss_ce * self.lambda_base
        return loss


class CoherenceLoss(nn.Module):
    """
    This class computes the coherence loss between the predicted meronym labels and the predicted base labels.

    The coherence loss encourages consistency between the meronym predictions and base label predictions.
    """
    def __init__(self, lambda_coh=1.0, epsilon=1e-8):
        """
        Initializes the CoherenceLoss.

        Parameters:
            lambda_coh: Weighting factor for the coherence loss.
        """
        super(CoherenceLoss, self).__init__()
        self.lambda_coh = lambda_coh
        self.epsilon = epsilon

    def forward(self, P_mero, P_base, P_base_given_mero):
        """
        Args:
            P_mero: Tensor of shape [batch_size, num_queries, num_meronym_classes + 1]
                - Predicted meronym label probabilities.
            P_base: Tensor of shape [batch_size, num_base_classes]
                - Predicted base label probabilities.
            P_base_given_mero: Tensor of shape [batch_size, num_meronym_classes + 1, num_base_classes]
                - Learned prior distribution of base labels given meronym labels.
        Returns:
            loss: Scalar tensor representing the coherence loss.
        """
        # Compute P_mero_base: [batch_size, num_base_classes]
        P_mero_base = torch.bmm(P_mero, P_base_given_mero).sum(dim=1)

        # Normalize P_mero_base to form a valid probability distribution
        P_mero_base = P_mero_base / (P_mero_base.sum(dim=1, keepdim=True) + self.epsilon)

        # Add eps to avoid log(0)
        P_base = P_base + self.epsilon
        P_mero_base = P_mero_base + self.epsilon

        # Compute KL divergences
        kl1 = F.kl_div(torch.log(P_base), P_mero_base, reduction='batchmean')
        kl2 = F.kl_div(torch.log(P_mero_base), P_base, reduction='batchmean')

        # Total coherence loss
        loss_coh = kl1 + kl2
        loss = loss_coh * self.lambda_coh

        return loss


class ToTLoss(nn.Module):
    def __init__(self, config, num_mero_classes):
        super(ToTLoss, self).__init__()
        matcher = HungarianMatcher()
        self.mero_loss = MeronymLoss(num_mero_classes, matcher, config.LOSS.LAMBDA_MERO, config.LOSS.EOS_COEF)
        self.base_loss = BaseLoss(config.LOSS.LAMBDA_BASE)
        self.coh_loss = CoherenceLoss(config.LOSS.LAMBDA_COH)

    def forward(self, outputs, targets):
        mero_logits = outputs['mero']
        base_logits = outputs['base']
        base_given_mero_logits = outputs['base_given_mero']

        P_mero = mero_logits.softmax(dim=-1)
        P_base = base_logits.softmax(dim=-1)
        P_base_given_mero = base_given_mero_logits.softmax(dim=-1)

        mero_labels = [torch.unique(targets["mero"][i]) for i in range(P_mero.shape[0])]
        base_labels = targets["base"]

        L_mero = self.mero_loss(mero_logits, mero_labels)
        L_base = self.base_loss(base_logits, base_labels)
        L_coh = self.coh_loss(P_mero, P_base, P_base_given_mero)
        L_total = L_mero + L_base + L_coh

        return {
            "total_loss": L_total,
            "mero_loss": L_mero,
            "base_loss": L_base,
            "coh_loss": L_coh
        }

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, tau):
        super(SupConLoss, self).__init__()
        self.tau = tau

    def _flatten_views(self, embeddings, labels):
        """
        Supports:
        - tuple/list of 2 tensors: (z1, z2), each [B, D]
        - tensor [B, V, D] (recommended for multi-view)
        - tensor [N, D] (legacy single-view format)
        Returns flattened embeddings [N_total, D] and expanded labels [N_total].
        """
        if isinstance(embeddings, (tuple, list)):
            if len(embeddings) < 2:
                raise ValueError("SupConLoss requires at least 2 views per sample.")
            embeddings = torch.cat(embeddings, dim=0)
            labels = labels.repeat(len(embeddings) // labels.shape[0])
            return embeddings, labels

        if embeddings.dim() == 3:
            bsz, n_views, feat_dim = embeddings.shape
            embeddings = embeddings.reshape(bsz * n_views, feat_dim)
            labels = labels.repeat_interleave(n_views)
            return embeddings, labels

        if embeddings.dim() == 2:
            return embeddings, labels

        raise ValueError(
            "Unsupported embeddings shape. Use (z1, z2), [B, V, D], or legacy [N, D]."
        )

    def forward(self, embeddings, labels):
        if labels is None:
            raise ValueError("SupConLoss is supervised and requires labels.")

        embeddings, labels = self._flatten_views(embeddings, labels)

        # L2-normalize so dot product becomes cosine similarity.
        embeddings = F.normalize(embeddings, dim=1)

        logits = torch.mm(embeddings, embeddings.T) / self.tau
        n = embeddings.shape[0]
        device = embeddings.device

        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_self = torch.eye(n, device=device)

        # Keep only non-self positive pairs.
        mask_pos = mask_pos * (1.0 - mask_self)

        # Numerical stabilization.
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        exp_logits = torch.exp(logits) * (1.0 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

        num_pos = mask_pos.sum(dim=1).clamp(min=1.0)
        loss = -(mask_pos * log_prob).sum(dim=1) / num_pos
        return loss.mean()
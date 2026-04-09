import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, tau):
        super(SupConLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings, labels):
        inner_product = torch.exp(embeddings[..., None] * embeddings[..., None]/ self.tau)
        #mask = labels[..., None] == labels[..., None].transpose(0, 1)
        # L2-normalize embeddings so dot product = cosine similarity
        embeddings = F.normalize(embeddings, dim=1)

        # Pairwise cosine similarity scaled by temperature (N, N)
        sim = torch.mm(embeddings, embeddings.T) / self.tau

        N = embeddings.shape[0]
        device = embeddings.device

        # Mask: positives = same class, excluding self
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_self = torch.eye(N, device=device)
        mask_pos = mask_pos - mask_self  # (N, N)

        # Numerical stability: subtract row-wise max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exponentiate, zero-out self-pairs
        exp_sim = torch.exp(sim) * (1 - mask_self)

        # log( exp(sim_ip) / sum_a!=i exp(sim_ia) ) = sim_ip - log(denom)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Average over positive pairs per anchor, then over batch
        num_pos = mask_pos.sum(dim=1).clamp(min=1)
        loss = -(mask_pos * log_prob).sum(dim=1) / num_pos

        return loss.mean()
import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, tau):
        super(SupConLoss, self).__init__()
        self.tau = tau

    def forward(self, embedding1, embedding2):
        inner_product = torch.exp(embedding1[..., None] * embedding2[..., None]/ self.tau)
        quotient = inner_product / torch.sum(inner_product, dim=-1, keepdim=True)
        loss = -torch.log(quotient)
        loss = torch.mean(loss)
        return loss
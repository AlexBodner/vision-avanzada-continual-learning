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
        loss_total = 0
        for i in range(embeddings.shape[0]):
            suma = 0
            suma_denominador = torch.sum(inner_product[i][:]) - inner_product[i][i]
            misma_clase = 0
            for j in range(embeddings.shape[0]):
                if labels[i] == labels[j] and i != j:
                    misma_clase += 1
                    suma += torch.exp(inner_product[i][j] / self.tau) / suma_denominador   
            
            loss_total += torch.log(suma/ misma_clase)
        return loss_total
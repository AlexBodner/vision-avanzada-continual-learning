import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, tau):
        super(SupConLoss, self).__init__()
        self.tau = tau

    def _flatten_views(self, embeddings, labels):
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
        embeddings = F.normalize(embeddings, dim=1)

        logits = torch.mm(embeddings, embeddings.T) / self.tau
        n = embeddings.shape[0]
        device = embeddings.device

        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_self = torch.eye(n, device=device)
        mask_pos = mask_pos * (1.0 - mask_self)

        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits) * (1.0 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

        num_pos = mask_pos.sum(dim=1).clamp(min=1.0)
        loss = -(mask_pos * log_prob).sum(dim=1) / num_pos
        return loss.mean()

class AsymmetricSupConLoss(nn.Module):
    """
    Asymmetric Supervised Contrastive Loss (Eq. 3 del paper Co2L).
    
    Para muestras de la tarea actual (current), usa etiquetas para formar pares positivos.
    Para muestras del buffer, solo usa su otra vista aumentada como par positivo.
    """
    def __init__(self, tau=0.07):
        super(AsymmetricSupConLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings, labels, num_current):
        if isinstance(embeddings, (tuple, list)):
            z = torch.cat(embeddings, dim=0)
        else:
            z = embeddings
            
        z = F.normalize(z, dim=1)
        n_total = z.shape[0]     
        bsz = n_total // 2       
        device = z.device
        labels_ext = labels.repeat(2)
        
        logits = torch.mm(z, z.T) / self.tau
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        mask_self = torch.eye(n_total, device=device)
        
        # SimCLR style
        indices = torch.arange(n_total, device=device)
        other_view_indices = (indices + bsz) % n_total
        mask_pos = torch.zeros((n_total, n_total), device=device)
        mask_pos[indices, other_view_indices] = 1.0
        
        # Current task positives
        current_indices = torch.cat([
            torch.arange(0, num_current, device=device),
            torch.arange(bsz, bsz + num_current, device=device)
        ])
        
        # Matriz de coincidencia de etiquetas
        label_match = (labels_ext.unsqueeze(0) == labels_ext.unsqueeze(1)).float()
        mask_pos[current_indices] = label_match[current_indices] * (1.0 - mask_self[current_indices])
        
        exp_logits = torch.exp(logits) * (1.0 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
        
        num_pos = mask_pos.sum(dim=1).clamp(min=1.0)
        loss = -(mask_pos * log_prob).sum(dim=1) / num_pos
        return loss.mean()

class IRDLoss(nn.Module):
    """
    Instance-wise Relation Distillation Loss (Eq. 4 del paper Co2L).
    
    Minimiza la divergencia KL entre las distribuciones de similitud del 
    modelo actual y el modelo previo (teacher).
    """
    def __init__(self, kappa=0.07, kappa_star=0.07):
        super(IRDLoss, self).__init__()
        self.kappa = kappa             
        self.kappa_star = kappa_star   

    def forward(self, student_z, teacher_z):
        student_z = F.normalize(student_z, dim=1)
        teacher_z = F.normalize(teacher_z, dim=1)
        
        n = student_z.shape[0]
        device = student_z.device
        mask_self = torch.eye(n, device=device)
        
        sim_s = torch.mm(student_z, student_z.T) / self.kappa
        sim_t = torch.mm(teacher_z, teacher_z.T) / self.kappa_star
        
        def masked_softmax(logits):
            logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
            exp_logits = torch.exp(logits) * (1.0 - mask_self)
            return exp_logits / exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12)

        p_s = masked_softmax(sim_s)
        p_t = masked_softmax(sim_t)
        
        loss = -(p_t * torch.log(p_s.clamp(min=1e-12))).sum(dim=1).mean()
        return loss
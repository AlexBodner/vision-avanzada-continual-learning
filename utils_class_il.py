import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_class_il(model, test_loader, device, active_classes):
    """
    Evalúa el modelo en el escenario Class-IL.
    El modelo debe predecir la clase correcta entre todas las active_classes.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x) # Se asume que el modelo ya tiene aplicada la máscara interna
            
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return 100. * correct / total

def compute_class_prototypes(model, train_loader, device):
    """  
    Calcula el embedding promedio (prototipo) de cada clase presente en el loader.
    Útil para clasificación NCM (Nearest Class Mean) en Co2L.
    """
    model.eval()
    prototypes = {} # {class_id: sum_embeddings}
    counts = {}     # {class_id: count}
    
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            # Obtenemos los embeddings del backbone (dim 32)
            embeddings = model.backbone(x)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            for i in range(x.size(0)):
                label = y[i].item()
                if label not in prototypes:
                    prototypes[label] = embeddings[i].clone()
                    counts[label] = 1
                else:
                    prototypes[label] += embeddings[i]
                    counts[label] += 1
                    
    # Promediar
    for label in prototypes:
        prototypes[label] = prototypes[label] / counts[label]
        prototypes[label] = F.normalize(prototypes[label], p=2, dim=0)
        
    return prototypes

def evaluate_ncm(model, test_loader, prototypes, device):
    """
    Evalúa usando Nearest Class Mean sobre los prototipos guardados.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Convertir dict de prototipos a tensor para cálculo rápido
    labels = sorted(prototypes.keys())
    proto_tensor = torch.stack([prototypes[l] for l in labels]).to(device)
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            embeddings = model.backbone(x)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Similitud coseno contra todos los prototipos
            # embeddings: [B, D], proto_tensor: [C, D] -> scores: [B, C]
            scores = torch.matmul(embeddings, proto_tensor.t())
            
            _, idx = scores.max(1)
            predicted = torch.tensor([labels[i] for i in idx]).to(device)
            
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return 100. * correct / total

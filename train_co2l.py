import os
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset

from models import CNN, Co2LModel
from dataloaders import SequentialCIFAR10
from losses import AsymmetricSupConLoss, IRDLoss
from utils import save_co2l_model, load_co2l_model

def train_co2l_phase1(model, teacher, train_loader, device, epochs=20, lr=1e-3, tau=0.07, lambda_ird=1.0):
    model.train()
    model.unfreeze_representation()
    
    optimizer = torch.optim.Adam(
        list(model.backbone.parameters()) + list(model.projection_head.parameters()), 
        lr=lr
    )
    
    criterion_con = AsymmetricSupConLoss(tau=tau)
    criterion_ird = IRDLoss(kappa=tau, kappa_star=tau)
    
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/{epochs}")
        
        for (x1, x2), y in pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            bsz = x1.shape[0]
            
            # En nuestro loader de Co2L, mezclamos tarea actual y buffer.
            # Por simplicidad y consistencia con el paper, calculamos num_current.
            # (En el paper esto suele ser el batch_size de la tarea actual).
            # Aquí usaremos el bsz completo como referencia de seguridad.
            num_current = bsz 
            
            # Forward estudiante
            z1 = model.forward_projection(x1)
            z2 = model.forward_projection(x2)
            
            loss_con = criterion_con((z1, z2), y, num_current=num_current)
            
            loss_ird = torch.tensor(0.0).to(device)
            if teacher is not None:
                with torch.no_grad():
                    t1 = teacher.forward_projection(x1)
                    t2 = teacher.forward_projection(x2)
                loss_ird = criterion_ird(torch.cat([z1, z2], dim=0), torch.cat([t1, t2], dim=0))
            
            loss = loss_con + lambda_ird * loss_ird
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "con": f"{loss_con.item():.4f}", 
                "ird": f"{loss_ird.item():.4f}"
            })
            
    return model

def train_co2l_phase2(
    model, 
    task_id, 
    train_loader, 
    device, 
    epochs=20, 
    lr=0.1
):
    """
    Fase 2: Entrenamiento del Clasificador Lineal.
    Backbone congelado. Solo se entrena la cabeza de la tarea actual.
    """
    model.eval()
    model.freeze_representation()
    
    if not model.classifier.has_task(task_id):
        model.classifier.add_task(task_id, num_classes=2)
    
    model.classifier.heads[str(task_id)].train()
    
    # Solo optimizar la cabeza actual (SGD como sugiere el paper para fase lineal)
    optimizer = torch.optim.SGD(
        model.classifier.heads[str(task_id)].parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Phase 2 - Task {task_id} - Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            logits = model.forward_classifier(x, task_id)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
            
    return model

def evaluate_co2l(model, test_loaders, device):
    """Evalúa el modelo en todas las tareas vistas hasta ahora."""
    model.eval()
    results = {}
    
    for task_id, loader in test_loaders.items():
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model.forward_classifier(x, task_id)
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        acc = 100. * correct / total
        results[task_id] = acc
        print(f"Accuracy Task {task_id}: {acc:.2f}%")
        
    return results

if __name__ == "__main__":
    # Configuración básica
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 128
    NUM_TASKS = 5
    EPOCHS_P1 = 10  # El paper usa más, pero 10 es un buen balance inicial
    EPOCHS_P2 = 20
    
    print(f"Iniciando Co2L en dispositivo: {device}")
    
    # Inicializar datos y modelo
    seq_cifar = SequentialCIFAR10(batch_size=BATCH_SIZE, buffer_size=200)
    backbone = CNN(in_channels=3, embedding_dim=32)
    model = Co2LModel(backbone, embedding_dim=32, proj_dim=128).to(device)
    
    teacher = None
    all_results = []
    
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*20} TAREA {task_id} {'='*20}")
        
        # 1. FASE 1: APRENDIZAJE DE REPRESENTACIÓN (CONTRASTIVO)
        # Usamos el loader de dos vistas con buffer (si task_id > 0)
        train_loader_p1 = seq_cifar.get_task_il_two_view_train_loader(task_id, use_buffer=(task_id > 0))
        model = train_co2l_phase1(model, teacher, train_loader_p1, device, epochs=EPOCHS_P1)
        
        # 2. FASE 2: ENTRENAMIENTO DE CABEZA LINEAL (CLASIFICACIÓN)
        # Usamos el loader estándar de la tarea actual (sin buffer para Phase 2)
        train_loader_p2 = seq_cifar.get_task_il_train_loader(task_id, use_buffer=False)
        model = train_co2l_phase2(model, task_id, train_loader_p2, device, epochs=EPOCHS_P2)
        
        # Guardar snapshot para la siguiente tarea (Teacher)
        teacher = deepcopy(model).eval()
        for param in teacher.parameters():
            param.requires_grad = False
            
        # 3. EVALUACIÓN TASK-IL
        test_loaders = seq_cifar.get_task_il_test_loaders(task_id)
        print(f"\nEvaluación final de Tarea {task_id}:")
        results = evaluate_co2l(model, test_loaders, device)
        all_results.append(results)
        
        # Actualizar buffer al final de la tarea para la siguiente
        seq_cifar.update_buffer(task_id)
        
        # Guardar checkpoint
        os.makedirs("checkpoints/co2l", exist_ok=True)
        save_co2l_model(model, f"checkpoints/co2l/task_{task_id}.pth")

    print("\n" + "="*50)
    print("ENTRENAMIENTO CO2L FINALIZADO")
    print("="*50)

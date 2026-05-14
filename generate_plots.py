import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
from models import CNN, Co2LModel
from dataloaders import SequentialCIFAR10
from utils import load_co2l_model

def generate_tsne_plot(task_id=0):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seq_cifar = SequentialCIFAR10(batch_size=128)
    
    # 1. Cargar modelo
    ckpt_path = f"checkpoints/co2l/task_{task_id}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: No se encontró el checkpoint en {ckpt_path}")
        return
        
    model = load_co2l_model(ckpt_path, device)
    model.eval()
    
    # 2. Obtener embeddings de test
    from torch.utils.data import DataLoader
    test_ds = seq_cifar.get_task_test_dataset(task_id)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            embeddings = model.backbone(x)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(y.numpy())
            
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    
    # 3. t-SNE
    print("Calculando t-SNE... (esto puede tardar un momento)")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # 4. Graficar
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='viridis', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=seq_cifar.task_classes[task_id])
    plt.title(f"Proyección t-SNE de Embeddings (Tarea {task_id})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("embeddings_tsne_task0.png")
    print("Gráfico guardado como embeddings_tsne_task0.png")

def generate_supcon_loss_plot():
    ckpt_path = "checkpoints/task_0_pretrain/supcon_backbone.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: No se encontró el checkpoint de SupCon en {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])

    if len(train_losses) == 0 or len(val_losses) == 0:
        print("Error: El checkpoint no contiene train_losses/val_losses.")
        return

    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', linewidth=2, label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, label='Validation Loss')
    plt.title("Evolución de la Pérdida SupCon (Tarea 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("supcon_loss_task0.png")
    print("Gráfico guardado como supcon_loss_task0.png")

def generate_task_il_curves():
    # Datos de evolución (promedio Task-IL acumulado) con pretraining SupCon de 30 épocas
    tasks = np.arange(5)

    # Co2L reentrenado (corrida final homogénea)
    co2l = [95.10, 88.55, 84.32, 82.04, 79.82]

    # Naive / EWC / LwF actualizados con las corridas más recientes
    naive = [91.80, 74.05, 69.90, 72.29, 75.64]
    ewc = [91.80, 74.60, 73.02, 73.53, 73.87]
    lwf = [91.80, 65.40, 64.28, 63.14, 64.49]
    
    plt.figure(figsize=(10, 6))
    plt.plot(tasks, co2l, marker='s', linewidth=3, label='Co2L', color='#d62728')
    plt.plot(tasks, naive, marker='o', linewidth=2, label='Naive FT', color='#7f7f7f', linestyle='--')
    plt.plot(tasks, ewc, marker='^', linewidth=2, label='EWC', color='#1f77b4')
    plt.plot(tasks, lwf, marker='v', linewidth=2, label='LwF', color='#ff7f0e')
    
    plt.title("Evolución de la Precisión Promedio (Task-IL)")
    plt.xlabel("Cantidad de Tareas Aprendidas")
    plt.ylabel("Accuracy Promedio (%)")
    plt.xticks(tasks, [f"Task {i}" for i in tasks])
    plt.ylim(50, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("taskil_accuracy_curves.png")
    print("Gráfico guardado como taskil_accuracy_curves.png")

if __name__ == "__main__":
    generate_supcon_loss_plot()
    generate_task_il_curves()
    generate_tsne_plot(0)

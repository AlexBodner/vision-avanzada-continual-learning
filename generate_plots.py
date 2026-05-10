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

def generate_dummy_loss_plot():
    # Basado en los logs reales que vimos de SupCon Task 0
    epochs = np.arange(1, 11)
    loss = [5.4189, 5.3379, 5.2856, 5.2479, 5.2394, 5.2070, 5.2030, 5.1840, 5.1715, 5.1673]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.title("Evolución de la Pérdida SupCon (Tarea 0)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("supcon_loss_task0.png")
    print("Gráfico guardado como supcon_loss_task0.png")

if __name__ == "__main__":
    generate_dummy_loss_plot()
    generate_tsne_plot(0)

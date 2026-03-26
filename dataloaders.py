"""
Dataloaders para Continual Learning sobre Seq-CIFAR-10.

Divide CIFAR-10 en N tareas secuenciales (por defecto 5, con 2 clases por tarea).
Soporta dos escenarios de evaluación:
  - Class-Incremental Learning (Class-IL): en test, clasificar entre TODAS las clases vistas.
  - Task-Incremental Learning (Task-IL): en test, clasificar solo entre las clases de una tarea dada.

Incluye un Replay Buffer con reservoir sampling para métodos que lo requieran.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from torchvision import datasets, transforms
from typing import List, Tuple, Optional, Dict
from copy import deepcopy


# ──────────────────────────────────────────────────────────────────────────────
# Configuración por defecto de Seq-CIFAR-10
# ──────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Orden de clases por tarea (2 clases por tarea, 5 tareas)
DEFAULT_TASK_CLASSES: List[List[int]] = [
    [0, 1],  # Tarea 0: airplane, automobile
    [2, 3],  # Tarea 1: bird, cat
    [4, 5],  # Tarea 2: deer, dog
    [6, 7],  # Tarea 3: frog, horse
    [8, 9],  # Tarea 4: ship, truck
]

# Transforms estándar para CIFAR-10
CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

CIFAR10_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrapper que filtra por clases y opcionalmente remapea labels
# ──────────────────────────────────────────────────────────────────────────────

class TaskDataset(Dataset):
    """
    Wrapper que filtra un dataset base por un subconjunto de clases.

    Args:
        base_dataset: Dataset original de CIFAR-10.
        class_ids: Lista de class IDs a incluir.
        remap_labels: Si True, remapea las labels a [0, 1, ..., len(class_ids)-1]
                      (útil para Task-IL). Si False, mantiene las labels originales
                      (útil para Class-IL).
    """

    def __init__(self, base_dataset: Dataset, class_ids: List[int],
                 remap_labels: bool = False):
        self.base_dataset = base_dataset
        self.class_ids = sorted(class_ids)
        self.remap_labels = remap_labels

        # Mapeo de label original → label local (0, 1, ...)
        self.label_map = {c: i for i, c in enumerate(self.class_ids)}

        # Filtrar índices donde el target pertenece a class_ids
        if hasattr(base_dataset, 'targets'):
            targets = np.array(base_dataset.targets)
        else:
            targets = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

        mask = np.isin(targets, self.class_ids)
        self.indices = np.where(mask)[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]
        if self.remap_labels:
            label = self.label_map[label]
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffer con Reservoir Sampling
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Buffer de memoria de tamaño fijo con reservoir sampling.

    Almacena ejemplos (imagen como tensor, label original) y los combina
    con datos de la tarea actual al entrenar.

    Args:
        max_size: Tamaño máximo del buffer.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.buffer: List[Tuple[torch.Tensor, int]] = []
        self._n_seen = 0  # Total de ejemplos vistos (para reservoir sampling)

    def __len__(self) -> int:
        return len(self.buffer)

    def update(self, dataset: Dataset):
        """
        Agrega ejemplos del dataset al buffer usando reservoir sampling.
        Llamar al final de cada tarea.
        """
        for i in range(len(dataset)):
            img, label = dataset[i]
            # Asegurar que img sea un tensor
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)

            self._n_seen += 1
            if len(self.buffer) < self.max_size:
                self.buffer.append((img.clone(), int(label)))
            else:
                # Reservoir sampling: reemplazar con probabilidad max_size / n_seen
                j = np.random.randint(0, self._n_seen)
                if j < self.max_size:
                    self.buffer[j] = (img.clone(), int(label))

    def get_dataset(self) -> Optional['BufferDataset']:
        """Retorna un Dataset con los ejemplos almacenados, o None si está vacío."""
        if len(self.buffer) == 0:
            return None
        return BufferDataset(self.buffer)

    def get_class_distribution(self) -> Dict[int, int]:
        """Retorna la distribución de clases en el buffer."""
        dist: Dict[int, int] = {}
        for _, label in self.buffer:
            dist[label] = dist.get(label, 0) + 1
        return dist


class BufferDataset(Dataset):
    """Dataset wrapper para los ejemplos almacenados en el ReplayBuffer."""

    def __init__(self, buffer: List[Tuple[torch.Tensor, int]]):
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int):
        img, label = self.buffer[idx]
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal: SequentialCIFAR10
# ──────────────────────────────────────────────────────────────────────────────

class SequentialCIFAR10:
    """
    Gestiona la división de CIFAR-10 en tareas secuenciales y provee
    dataloaders para entrenamiento y evaluación en escenarios Class-IL y Task-IL.

    Args:
        data_root: Directorio donde descargar/cargar CIFAR-10.
        n_tasks: Número de tareas (por defecto 5).
        task_classes: Lista de listas con los class IDs por tarea.
                      Si es None, usa DEFAULT_TASK_CLASSES.
        train_transform: Transform para datos de entrenamiento.
        test_transform: Transform para datos de test.
        batch_size: Tamaño de batch para dataloaders.
        num_workers: Workers para carga de datos.
        buffer_size: Tamaño del replay buffer (0 = sin buffer).
    """

    def __init__(
        self,
        data_root: str = "./data",
        n_tasks: int = 5,
        task_classes: Optional[List[List[int]]] = None,
        train_transform=None,
        test_transform=None,
        batch_size: int = 32,
        num_workers: int = 2,
        buffer_size: int = 200,
        val_split: float = 0.1,
    ):
        self.data_root = data_root
        self.n_tasks = n_tasks
        self.task_classes = task_classes or DEFAULT_TASK_CLASSES[:n_tasks]
        self.train_transform = train_transform or CIFAR10_TRAIN_TRANSFORM
        self.test_transform = test_transform or CIFAR10_TEST_TRANSFORM
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert 0.0 <= val_split < 1.0, "val_split debe estar en [0, 1)"
        self.val_split = val_split

        assert len(self.task_classes) == n_tasks, \
            f"task_classes tiene {len(self.task_classes)} tareas, se esperaban {n_tasks}"

        # Descargar CIFAR-10
        self.train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=self.train_transform,
        )
        self.test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=self.test_transform,
        )

        # Replay buffer
        self.buffer = ReplayBuffer(max_size=buffer_size) if buffer_size > 0 else None

        # Acumulador de clases vistas (para Class-IL)
        self.classes_seen: List[int] = []

    # ── Datasets por tarea ────────────────────────────────────────────────

    def get_task_train_dataset(self, task_id: int, remap_labels: bool = False) -> TaskDataset:
        """Dataset de entrenamiento para una tarea específica (sin replay)."""
        classes = self.task_classes[task_id]
        return TaskDataset(self.train_dataset, classes, remap_labels=remap_labels)

    def _split_train_val(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Divide un dataset en train y validación de forma determinista.
        Usa val_split como fracción del total.
        """
        n_total = len(dataset)
        n_val = max(1, int(n_total * self.val_split))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(42)
        return random_split(dataset, [n_train, n_val], generator=generator)

    def get_task_test_dataset(self, task_id: int, remap_labels: bool = False) -> TaskDataset:
        """Dataset de test para una tarea específica."""
        classes = self.task_classes[task_id]
        return TaskDataset(self.test_dataset, classes, remap_labels=remap_labels)

    # ── Dataloaders para ENTRENAMIENTO ────────────────────────────────────

    def get_train_val_loaders(
        self, task_id: int, use_buffer: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Retorna (train_loader, val_loader) para Class-IL.

        El split se hace sobre los datos de la tarea actual antes de combinar
        con el replay buffer (el buffer solo va al train split).

        Args:
            task_id: ID de la tarea actual (0-indexed).
            use_buffer: Si True, agrega el replay buffer al train loader.
        """
        task_ds = self.get_task_train_dataset(task_id, remap_labels=False)
        train_ds, val_ds = self._split_train_val(task_ds)

        if use_buffer and self.buffer is not None:
            buffer_ds = self.buffer.get_dataset()
            if buffer_ds is not None:
                train_ds = ConcatDataset([train_ds, buffer_ds])

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def get_task_il_train_val_loaders(
        self, task_id: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Retorna (train_loader, val_loader) para Task-IL.
        Labels remapeadas a [0, 1] (locales a la tarea).
        """
        task_ds = self.get_task_train_dataset(task_id, remap_labels=True)
        train_ds, val_ds = self._split_train_val(task_ds)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def get_train_loader(self, task_id: int, use_buffer: bool = False) -> DataLoader:
        """
        Dataloader de entrenamiento para Class-IL.

        Labels originales [0-9], el modelo debe aprender a clasificar
        entre todas las clases vistas hasta ahora.

        Args:
            task_id: ID de la tarea actual (0-indexed).
            use_buffer: Si True, combina datos de la tarea con el replay buffer.
        """
        task_ds = self.get_task_train_dataset(task_id, remap_labels=False)
        dataset_list = [task_ds]

        if use_buffer and self.buffer is not None:
            buffer_ds = self.buffer.get_dataset()
            if buffer_ds is not None:
                dataset_list.append(buffer_ds)

        combined = ConcatDataset(dataset_list) if len(dataset_list) > 1 else dataset_list[0]

        return DataLoader(
            combined,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def get_task_il_train_loader(self, task_id: int, use_buffer: bool = False) -> DataLoader:
        """
        Dataloader de entrenamiento para Task-IL.

        Labels remapeadas a [0, 1] (i.e., locales a la tarea).
        Mismo dataset pero con remap_labels=True.

        Nota: el replay buffer conserva labels originales; si se usa junto
        con Task-IL, se debe tener cuidado de remapear en el loop de
        entrenamiento según el task_id correspondiente.
        """
        task_ds = self.get_task_train_dataset(task_id, remap_labels=True)

        return DataLoader(
            task_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ── Dataloaders para EVALUACIÓN ───────────────────────────────────────

    def get_class_il_test_loader(self, up_to_task: int) -> DataLoader:
        """
        Dataloader de evaluación Class-IL.

        Incluye datos de test de TODAS las tareas vistas hasta `up_to_task`
        (inclusive). Labels originales [0-9] → el modelo debe distinguir
        entre todas las clases sin saber la tarea.

        Args:
            up_to_task: Evaluar tareas 0, 1, ..., up_to_task.
        """
        all_classes = []
        for t in range(up_to_task + 1):
            all_classes.extend(self.task_classes[t])

        combined_test = TaskDataset(self.test_dataset, all_classes, remap_labels=False)

        return DataLoader(
            combined_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_task_il_test_loaders(self, up_to_task: int) -> Dict[int, DataLoader]:
        """
        Dataloaders de evaluación Task-IL.

        Retorna un diccionario {task_id: DataLoader} para cada tarea vista.
        Cada loader contiene solo las clases de esa tarea con labels
        remapeadas a [0, 1, ..., classes_per_task - 1].

        Para evaluar Task-IL, se itera sobre cada task_id, se selecciona
        la cabeza/máscara correspondiente, y se evalúa solo con sus 2 clases.

        Args:
            up_to_task: Evaluar tareas 0, 1, ..., up_to_task.
        """
        loaders = {}
        for t in range(up_to_task + 1):
            task_test = self.get_task_test_dataset(t, remap_labels=True)
            loaders[t] = DataLoader(
                task_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return loaders

    # ── Gestión del buffer ────────────────────────────────────────────────

    def update_buffer(self, task_id: int):
        """
        Actualiza el replay buffer con datos de la tarea actual.
        Llamar al final de cada tarea.

        Nota: usa el dataset base SIN augmentations para guardar
        representaciones limpias en el buffer. Las imágenes se almacenan
        como tensores normalizados (con el test_transform).
        """
        if self.buffer is None:
            return

        # Crear dataset con test_transform para guardar imágenes limpias
        clean_train = datasets.CIFAR10(
            root=self.data_root, train=True, download=False,
            transform=self.test_transform,
        )
        classes = self.task_classes[task_id]
        task_ds = TaskDataset(clean_train, classes, remap_labels=False)
        self.buffer.update(task_ds)

    def update_classes_seen(self, task_id: int):
        """Registra las clases de la tarea como vistas (para tracking)."""
        for c in self.task_classes[task_id]:
            if c not in self.classes_seen:
                self.classes_seen.append(c)

    # ── Utilidades ────────────────────────────────────────────────────────

    def get_task_class_names(self, task_id: int) -> List[str]:
        """Retorna los nombres de las clases de una tarea."""
        return [CIFAR10_CLASSES[c] for c in self.task_classes[task_id]]

    def get_all_seen_classes(self, up_to_task: int) -> List[int]:
        """Retorna todas las class IDs vistas hasta `up_to_task`."""
        classes = []
        for t in range(up_to_task + 1):
            classes.extend(self.task_classes[t])
        return sorted(classes)

    def get_n_classes_per_task(self) -> int:
        """Número de clases por tarea (asume uniforme)."""
        return len(self.task_classes[0])

    def get_total_classes(self, up_to_task: int) -> int:
        """Total de clases vistas hasta la tarea up_to_task."""
        return len(self.get_all_seen_classes(up_to_task))

    def summary(self):
        """Imprime un resumen de la configuración."""
        print(f"Seq-CIFAR-10 — {self.n_tasks} tareas")
        print(f"Buffer size: {self.buffer.max_size if self.buffer else 0}")
        print("-" * 45)
        for t in range(self.n_tasks):
            names = self.get_task_class_names(t)
            train_ds = self.get_task_train_dataset(t)
            test_ds = self.get_task_test_dataset(t)
            print(f"  Tarea {t}: {names}  "
                  f"(train: {len(train_ds)}, test: {len(test_ds)})")
        print("-" * 45)


# ──────────────────────────────────────────────────────────────────────────────
# Ejemplo de uso y loop de entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Crear el gestor de datasets
    seq_cifar = SequentialCIFAR10(
        data_root="./data",
        n_tasks=5,
        batch_size=32,
        buffer_size=200,
    )
    seq_cifar.summary()

    print("\n=== Simulación del loop de entrenamiento ===\n")

    for task_id in range(seq_cifar.n_tasks):
        print(f"--- Tarea {task_id}: {seq_cifar.get_task_class_names(task_id)} ---")

        # 1) Dataloaders de entrenamiento/validación (Class-IL)
        train_loader, val_loader = seq_cifar.get_train_val_loaders(task_id, use_buffer=True)
        print(f"  Train batches (Class-IL): {len(train_loader)}, "
              f"samples: {len(train_loader.dataset)}")
        print(f"  Val   batches (Class-IL): {len(val_loader)}, "
              f"samples: {len(val_loader.dataset)}")

        # 2) Dataloaders de entrenamiento/validación (Task-IL)
        task_il_train_loader, task_il_val_loader = seq_cifar.get_task_il_train_val_loaders(task_id)
        print(f"  Train batches (Task-IL):  {len(task_il_train_loader)}, "
              f"samples: {len(task_il_train_loader.dataset)}")
        print(f"  Val   batches (Task-IL):  {len(task_il_val_loader)}, "
              f"samples: {len(task_il_val_loader.dataset)}")

        # Simular un batch
        imgs, labels = next(iter(train_loader))
        print(f"  Batch shape: {imgs.shape}, labels: {labels.unique().tolist()}")

        # 3) Actualizar buffer y clases vistas al final de la tarea
        seq_cifar.update_buffer(task_id)
        seq_cifar.update_classes_seen(task_id)

        if seq_cifar.buffer:
            dist = seq_cifar.buffer.get_class_distribution()
            print(f"  Buffer: {len(seq_cifar.buffer)} ejemplos, distribución: {dist}")

        # 4) Evaluación Class-IL: test sobre todas las clases vistas
        class_il_test_loader = seq_cifar.get_class_il_test_loader(up_to_task=task_id)
        print(f"  Test Class-IL: {len(class_il_test_loader.dataset)} samples, "
              f"clases: {seq_cifar.get_all_seen_classes(task_id)}")

        # 5) Evaluación Task-IL: test por tarea separado
        task_il_test_loaders = seq_cifar.get_task_il_test_loaders(up_to_task=task_id)
        for t, loader in task_il_test_loaders.items():
            print(f"  Test Task-IL tarea {t}: {len(loader.dataset)} samples, "
                  f"clases locales [0, 1]")

        print()

# TP1 - Continual Learning en Seq-CIFAR-10

En este repositorio dejamos todo el codigo que usamos para el trabajo practico de aprendizaje continuo sobre Seq-CIFAR-10.

Incluye:

- preentrenamiento contrastivo supervisado en la tarea 0
- evaluacion incremental en Task-IL
- evaluacion incremental en Class-IL
- comparacion entre Naive Fine-tuning, EWC, LwF y Co2L
- generacion de figuras y reporte final

## Como esta organizado el repo

### Notebooks principales

- `supervised_contrastative_learning_42.ipynb`: entrenamos el backbone con SupCon en la tarea 0 y despues hacemos linear probe.
- `task_incremental_learning.ipynb`: corremos Task-IL para Naive y EWC.
- `lwf_task_incremental.ipynb`: corremos Task-IL para LwF.
- `co2l_task_incremental_plan.ipynb`: corremos Co2L en dos fases para las cinco tareas.
- `naive_class_il.ipynb`: evaluacion Class-IL para Naive.
- `ewc_class_il.ipynb`: evaluacion Class-IL para EWC.
- `lwf_class_il.ipynb`: evaluacion Class-IL para LwF.
- `co2l_class_il.ipynb`: evaluacion Class-IL para Co2L (comparando NCM vs Logits agregados).

### Donde implementamos cada parte principal

- Preentrenamiento SupCon + linear probe: `supervised_contrastative_learning_42.ipynb`.
- Task-IL con Naive y EWC: `task_incremental_learning.ipynb`.
- Task-IL con LwF: `lwf_task_incremental.ipynb`.
- Task-IL con Co2L (dos fases): `co2l_task_incremental_plan.ipynb`.
- Class-IL con Naive/EWC/LwF: `naive_class_il.ipynb`, `ewc_class_il.ipynb`, `lwf_class_il.ipynb`.
- Class-IL para Co2L (NCM y Logits): `co2l_class_il.ipynb` y funciones de `utils_class_il.py`.
- Carga de datos, particion en tareas y replay buffer: `dataloaders.py`.
- Definicion de modelos (backbone, cabezas y wrappers de CL): `models.py`.
- Definicion de perdidas (incluyendo Co2L): `losses.py`.
- Utilidades de entrenamiento/evaluacion para Task-IL: `train.py` y `train_ewc.py`.
- Entrenamiento de Co2L en formato script: `train_co2l.py`.
- Guardado/carga de checkpoints: `utils.py`.
- Generacion de figuras del informe: `generate_plots.py`.

### Modulos de soporte

- `dataloaders.py`: define `SequentialCIFAR10`, particiona CIFAR-10 en 5 tareas y maneja replay buffer.
- `models.py`: backbone CNN, cabezas lineales y modelos usados en Task-IL, Class-IL y Co2L.
- `losses.py`: perdidas contrastivas y de distillation para Co2L.
- `train.py`: utilidades de entrenamiento/evaluacion para clasificadores estandar en Task-IL.
- `train_ewc.py`: utilidades de entrenamiento/evaluacion adaptadas a EWC.
- `train_co2l.py`: version script del entrenamiento de Co2L.
- `utils.py`: helpers para guardar y cargar checkpoints.
- `utils_class_il.py`: helpers de Class-IL, incluyendo NCM para Co2L.
- `generate_plots.py`: genera las figuras finales del informe.

### Artefactos y salidas

- `checkpoints/`: pesos guardados de SupCon, Naive, EWC, LwF y Co2L.
- `overleaf_figures/`: copias de las figuras usadas en el informe.
- `data/cifar-10-batches-py/`: dataset CIFAR-10 usado en el trabajo.
- `informe_tecnico_tp1.tex`: informe en LaTeX.
- `I309_TP1.pdf`: version PDF del informe.

## Como correrlo

### Ejecucion con notebooks

Para reproducir lo que hicimos, hay que abrir los notebooks y ejecutarlos de arriba hacia abajo en este orden:

1. `supervised_contrastative_learning_42.ipynb`
2. `task_incremental_learning.ipynb`
3. `lwf_task_incremental.ipynb`
4. `naive_class_il.ipynb`
5. `ewc_class_il.ipynb`
6. `lwf_class_il.ipynb`
7. `co2l_task_incremental_plan.ipynb`
8. `co2l_class_il.ipynb`
9. `python generate_plots.py`

### Scripts que se pueden ejecutar directo

No todos los `.py` estan pensados para correrse solos.

Se pueden ejecutar directamente:

- `python train_co2l.py`: corre una version script de Co2L.
- `python generate_plots.py`: regenera las figuras finales.

`train.py` y `train_ewc.py` los usamos como modulos auxiliares desde notebooks.

## Entorno sugerido

Nosotros corrimos el trabajo en Python con PyTorch. Para recrear un entorno desde cero:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib scikit-learn tqdm jupyter
```

En macOS con Apple Silicon, PyTorch puede usar `mps` automaticamente si esta disponible.

## Nota sobre Class-IL y NCM

- En Task-IL el modelo conoce la tarea en evaluacion y por eso podemos seleccionar la cabeza correspondiente.
- En Class-IL esa informacion no esta disponible.
- Para Co2L, la evaluacion de Class-IL la hicimos con NCM (`Nearest Class Mean`) sobre embeddings del backbone, porque nuestra implementacion principal del metodo esta orientada a aprender buena representacion compartida mas que una unica cabeza global de clases.

## Resultados finales reportados

- Naive Task-IL: 75.64
- EWC Task-IL: 73.87
- LwF Task-IL: 64.49
- Co2L Task-IL: 79.82
- Co2L Class-IL con NCM: 50.82
- Co2L Class-IL sin NCM: ~17% 

from copy import deepcopy

from torch import nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 8 * 8, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
   
class LinearProbe(nn.Module):
    def __init__(self, backbone, embedding_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class TaskIncrementalClassifier(nn.Module):
    def __init__(self, backbone, embedding_dim):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.heads = nn.ModuleDict()

    @staticmethod
    def  from_linear_model(backbone, embedding_dim, linear_head, task_id ):
        model = TaskIncrementalClassifier(backbone, embedding_dim)
        model.heads[task_id] = deepcopy(linear_head)
        return model

    def task_ids(self):
        return (int(task_id) for task_id in self.heads.keys())
    def add_task(self, task_id, num_classes):
        task_key = str(task_id)
        if task_key in self.heads:
            raise ValueError(f"Task head {task_id} already exists.")
        head = nn.Linear(self.embedding_dim, num_classes)
        # Place the new head on the same device as the backbone
        device = next(self.backbone.parameters()).device
        self.heads[task_key] = head.to(device)

    def has_task(self, task_id):
        return str(task_id) in self.heads

    def get_head(self, task_id):
        task_key = str(task_id)
        if task_key not in self.heads:
            raise KeyError(f"Task head {task_id} does not exist.")
        return self.heads[task_key]

    def task_ids(self):
        return sorted(int(task_id) for task_id in self.heads.keys())

    def freeze_heads_except(self, task_id):
        active_task = str(task_id)
        for head_task, head in self.heads.items():
            requires_grad = head_task == active_task
            for param in head.parameters():
                param.requires_grad_(requires_grad)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)
    def forward(self, x, task_id):
        features = self.backbone(x)
        return self.get_head(task_id)(features)
    
    def forward_probs(self, x, task_id):
        logits = self.forward(x, task_id)
        return F.softmax(logits, dim=1)
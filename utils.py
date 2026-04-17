
import torch
from models import CNN, LinearProbe, TaskIncrementalClassifier

def load_classifier(checkpoint_path, device):
    head_ckpt = torch.load(f"{checkpoint_path}/linear_probe_head.pt", map_location=device)
    backbone_ckpt = torch.load(f"{checkpoint_path}/supcon_backbone.pt", map_location=device)

    # Recreate backbone model object, then load its state_dict
    backbone_model = CNN(in_channels=3, embedding_dim=32).to(device)
    backbone_model.load_state_dict(backbone_ckpt["model_state_dict"])
    backbone_model.eval()

    # Recreate linear probe with backbone model and load linear head
    reloaded_linear_probe = LinearProbe(
        backbone_model,
        embedding_dim=32,
        num_classes=head_ckpt["num_classes"],
    ).to(device)
    reloaded_linear_probe.classifier.load_state_dict(head_ckpt["classifier_state_dict"])
    reloaded_linear_probe.eval()

    print("Backbone + linear head loaded successfully into reloaded_linear_probe.")
    return reloaded_linear_probe


def load_task_incremental_from_pretrain(checkpoint_path, device, task_id=0):
    head_ckpt = torch.load(f"{checkpoint_path}/linear_probe_head.pt", map_location=device)
    backbone_ckpt = torch.load(f"{checkpoint_path}/supcon_backbone.pt", map_location=device)

    embedding_dim = backbone_ckpt.get("embedding_dim", 32)
    backbone_model = CNN(in_channels=3, embedding_dim=embedding_dim)
    backbone_model.load_state_dict(backbone_ckpt["model_state_dict"])

    model = TaskIncrementalClassifier(backbone_model, embedding_dim=embedding_dim)
    model.add_task(task_id=task_id, num_classes=head_ckpt["num_classes"])
    model.get_head(task_id).load_state_dict(head_ckpt["classifier_state_dict"])
    model = model.to(device)

    print("Backbone + task head loaded successfully into task-incremental classifier.")
    return model


def save_task_incremental_classifier(model, checkpoint_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "task_out_dims": {
                int(task_id): head.out_features
                for task_id, head in model.heads.items()
            },
        },
        checkpoint_path,
    )


def load_task_incremental_classifier(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_dim = checkpoint["embedding_dim"]

    backbone_model = CNN(in_channels=3, embedding_dim=embedding_dim)
    model = TaskIncrementalClassifier(backbone_model, embedding_dim=embedding_dim)

    for task_id, num_classes in checkpoint["task_out_dims"].items():
        model.add_task(task_id=int(task_id), num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model
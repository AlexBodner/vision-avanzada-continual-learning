import torch
from models import CNN, LinearProbe, TaskIncrementalClassifier, Co2LModel

def load_classifier(checkpoint_path, device):
    head_ckpt = torch.load(f"{checkpoint_path}/linear_probe_head.pt", map_location=device)
    backbone_ckpt = torch.load(f"{checkpoint_path}/supcon_backbone.pt", map_location=device)

    backbone_model = CNN(in_channels=3, embedding_dim=32).to(device)
    backbone_model.load_state_dict(backbone_ckpt["model_state_dict"])
    backbone_model.eval()

    reloaded_linear_probe = LinearProbe(
        backbone_model,
        embedding_dim=32,
        num_classes=head_ckpt["num_classes"],
    ).to(device)
    reloaded_linear_probe.classifier.load_state_dict(head_ckpt["classifier_state_dict"])
    reloaded_linear_probe.eval()

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


def save_co2l_model(model, checkpoint_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.classifier.embedding_dim,
            "task_out_dims": {
                int(task_id): head.out_features
                for task_id, head in model.classifier.heads.items()
            },
        },
        checkpoint_path,
    )


def load_co2l_model(checkpoint_path, device, proj_dim=128):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_dim = checkpoint["embedding_dim"]

    backbone = CNN(in_channels=3, embedding_dim=embedding_dim)
    model = Co2LModel(backbone, embedding_dim=embedding_dim, proj_dim=proj_dim)

    for task_id, num_classes in checkpoint["task_out_dims"].items():
        if not model.classifier.has_task(task_id):
            model.classifier.add_task(task_id=int(task_id), num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model
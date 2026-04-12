
import torch
from models import CNN, LinearProbe

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
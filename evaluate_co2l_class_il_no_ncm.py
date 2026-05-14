import torch

from dataloaders import SequentialCIFAR10
from utils import load_co2l_model


def evaluate_co2l_class_il_no_ncm(
    checkpoint_path="checkpoints/co2l/task_4.pth",
    data_root="./data",
    batch_size=128,
    up_to_task=4,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = load_co2l_model(checkpoint_path, device=device)
    model.eval()

    seq = SequentialCIFAR10(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        buffer_size=500,
    )
    test_loader = seq.get_class_il_test_loader(up_to_task=up_to_task)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            feats = model.backbone(x)
            logits_global = torch.full((x.size(0), 10), -1e9, device=device)

            for t in model.classifier.task_ids():
                head = model.classifier.get_head(t)
                local_logits = head(feats)
                class_ids = seq.task_classes[t]
                logits_global[:, class_ids[0]] = local_logits[:, 0]
                logits_global[:, class_ids[1]] = local_logits[:, 1]

            pred = logits_global.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    acc = 100.0 * correct / total
    print(f"Co2L Class-IL sin NCM (logits agregados): {acc:.2f}%")
    return acc


if __name__ == "__main__":
    evaluate_co2l_class_il_no_ncm()

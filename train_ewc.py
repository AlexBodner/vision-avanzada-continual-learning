from tqdm import tqdm
import torch
def train_classifier(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    num_epochs=10,
    device="mps",
    **kwargs
):
    train_losses = []
    val_losses = []
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training"):
        running_loss = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            loss = criterion(images, labels, model, **kwargs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
        train_losses.append(running_loss / len(train_dataloader))

        model.eval()
        running_val_loss = 0.0
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            loss = criterion(images, labels, model, **kwargs)
            running_val_loss += loss.item()

        val_losses.append(running_val_loss / len(val_dataloader))

    return train_losses, val_losses

def evaluate_classifier(model, split_dataloader, criterion, device, **kwargs):
    model.eval()
    split_running_loss = 0.0
    split_correct = 0
    split_total = 0

    # with torch.no_grad():
    #     for images, labels in split_dataloader:
    #         images = images.to(device)
    #         labels = labels.to(device)

    #         logits = model(images)
    #         split_loss = criterion(images, labels, model, **kwargs)

    #         split_running_loss += split_loss.item()
    #         preds = logits.argmax(dim=1)
    #         split_correct += (preds == labels).sum().item()
    #         split_total += labels.size(0)

    split_loss = split_running_loss / len(split_dataloader)
    split_acc = split_correct / split_total
    return split_loss, split_acc


def evaluate_task_incremental(model, task_id, loaders, criterion, device,**kwargs):
    task_results = {}
    for eval_task in range(0, task_id + 1):
        eval_loader = loaders[eval_task]
        eval_loss, eval_acc = evaluate_classifier(model, eval_loader, criterion, device, **kwargs)
        task_results[f"task_{eval_task}"] = {"loss": eval_loss, "acc": eval_acc}
    return task_results
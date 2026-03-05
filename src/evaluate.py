# src/evaluate.py

from sklearn.metrics import classification_report


def evaluate_model(model, dataloader, device):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            label = batch["label"].to(device)
            task = batch["task"][0]

            logits = model(input_ids, attention_mask, target_mask, task)
            prediction = logits.argmax(dim=1)

            preds.extend(prediction.cpu().tolist())
            labels.extend(label.cpu().tolist())

    print(classification_report(labels, preds))
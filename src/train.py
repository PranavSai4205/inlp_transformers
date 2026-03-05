# src/train.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DEVICE, LR, EPOCHS
from evaluate import evaluate_model


def train_model(model, qa_dataset, stance_dataset):
    model.to(DEVICE)

    qa_loader = DataLoader(qa_dataset, batch_size=16, shuffle=True)
    stance_loader = DataLoader(stance_dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()

        print(f"Epoch {epoch+1}")

        # Alternate training
        for qa_batch, stance_batch in zip(qa_loader, stance_loader):

            # QA batch
            input_ids = qa_batch["input_ids"].to(DEVICE)
            attention_mask = qa_batch["attention_mask"].to(DEVICE)
            target_mask = qa_batch["target_mask"].to(DEVICE)
            labels = qa_batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, target_mask, "qa")
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            # Stance batch
            input_ids = stance_batch["input_ids"].to(DEVICE)
            attention_mask = stance_batch["attention_mask"].to(DEVICE)
            target_mask = stance_batch["target_mask"].to(DEVICE)
            labels = stance_batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, target_mask, "stance")
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        print("Training epoch complete")
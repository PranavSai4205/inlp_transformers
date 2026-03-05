# local_sanity_check.py

import torch
from torch.utils.data import DataLoader
from src.datasets import QAEvasionDataset, PStanceDataset
from src.model import TargetAwareModel
from config import DEVICE

print("Loading datasets...")

qa_dataset = QAEvasionDataset("data/raw/QAEvasion.csv")
stance_dataset = PStanceDataset("data/raw/raw_train_trump.csv")  # change path if needed

qa_loader = DataLoader(qa_dataset, batch_size=4, shuffle=True)
stance_loader = DataLoader(stance_dataset, batch_size=4, shuffle=True)

print("Initializing model...")
model = TargetAwareModel()
model.to(DEVICE)

# -----------------------------------------
# 1️⃣ Forward Pass Test (QA)
# -----------------------------------------

print("\nTesting QA batch...")

qa_batch = next(iter(qa_loader))

input_ids = qa_batch["input_ids"].to(DEVICE)
attention_mask = qa_batch["attention_mask"].to(DEVICE)
target_mask = qa_batch["target_mask"].to(DEVICE)

logits = model(input_ids, attention_mask, target_mask, task="qa")

print("QA logits shape:", logits.shape)  # Expected: (4, 3)

# -----------------------------------------
# 2️⃣ Forward Pass Test (Stance)
# -----------------------------------------

print("\nTesting Stance batch...")

stance_batch = next(iter(stance_loader))

input_ids = stance_batch["input_ids"].to(DEVICE)
attention_mask = stance_batch["attention_mask"].to(DEVICE)
target_mask = stance_batch["target_mask"].to(DEVICE)

logits = model(input_ids, attention_mask, target_mask, task="stance")

print("Stance logits shape:", logits.shape)  # Expected: (4, 2)

# -----------------------------------------
# 3️⃣ Loss Test
# -----------------------------------------

print("\nTesting loss computation...")

loss_fn = torch.nn.CrossEntropyLoss()

labels = qa_batch["label"].to(DEVICE)
loss = loss_fn(logits, labels[:4])  # dummy slice to match shape
print("Loss computed successfully:", loss.item())

print("\nAll sanity checks passed.")
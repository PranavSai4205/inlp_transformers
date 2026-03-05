# src/datasets.py

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from config import MODEL_NAME, MAX_LEN


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class QAEvasionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        # Map hierarchical evasion labels
        def map_evasion_label(label_str):
            label_str = str(label_str)
            if label_str.startswith("1"):
                return 0  # Non-evasive
            elif label_str.startswith("2"):
                return 1  # Partially evasive
            elif label_str.startswith("3"):
                return 2  # Evasive
            else:
                return -1  # invalid

        # Create mapped column
        self.data["mapped_label"] = self.data["label"].apply(map_evasion_label)

        # Remove invalid rows
        self.data = self.data[self.data["mapped_label"] != -1]

        # Now extract fields AFTER filtering
        self.questions = self.data["interview_question"].tolist()
        self.answers = self.data["interview_answer"].tolist()
        self.labels = self.data["mapped_label"].tolist()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        question = str(self.questions[idx])
        answer = str(self.answers[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = tokenizer(
            question,
            answer,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create target mask (tokens before first [SEP])
        sep_index = (input_ids == tokenizer.sep_token_id).nonzero()[0].item()
        target_mask = torch.zeros_like(input_ids)
        target_mask[:sep_index] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "label": label,
            "task": "qa"
        }


class PStanceDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        label_map = {
            "AGAINST": 0,
            "FAVOR": 1
        }

        self.targets = self.data["Target"].tolist()
        self.texts = self.data["Tweet"].tolist()
        self.labels = self.data["Stance"].map(label_map).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        target = str(self.targets[idx])
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = tokenizer(
            target,
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        sep_index = (input_ids == tokenizer.sep_token_id).nonzero()[0].item()
        target_mask = torch.zeros_like(input_ids)
        target_mask[:sep_index] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "label": label,
            "task": "stance"
        }
# config.py
import torch
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
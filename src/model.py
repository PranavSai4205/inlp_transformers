# src/model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME


class TargetAwareModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        # Gating layer
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

        # Heads
        self.evasion_head = nn.Linear(hidden_size, 3)
        self.stance_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, target_mask, task):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state  # (B, L, H)

        # Compute target representation
        target_mask = target_mask.unsqueeze(-1)
        target_embeddings = hidden_states * target_mask
        target_sum = target_embeddings.sum(dim=1)
        target_count = target_mask.sum(dim=1).clamp(min=1)
        v_t = target_sum / target_count

        # Expand target vector
        v_t_expanded = v_t.unsqueeze(1).expand_as(hidden_states)

        # Concatenate token with target
        concat = torch.cat([hidden_states, v_t_expanded], dim=-1)

        # Compute gating
        gate_values = torch.sigmoid(self.gate(concat))
        gated_hidden = gate_values * hidden_states

        # Use gated CLS
        cls_rep = gated_hidden[:, 0, :]

        if task == "qa":
            return self.evasion_head(cls_rep)

        elif task == "stance":
            return self.stance_head(cls_rep)
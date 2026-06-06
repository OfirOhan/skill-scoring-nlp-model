"""
model.py – RoBERTa + MLP + CORAL head for ordinal regression (1-5 scoring).
"""

import torch
import torch.nn as nn
from transformers import RobertaModel

import config


class ScoringModel(nn.Module):
    """
    Architecture
    ────────────
    1. RoBERTa-base  →  [CLS] embedding  (768-d)
    2. MLP head:
         Linear(768 → 256) + ReLU + Dropout
         Linear(256 →  64) + ReLU + Dropout
         Linear( 64 →   4) + Sigmoid   ← CORAL cumulative probabilities
    """

    def __init__(self):
        super().__init__()

        # ── Transformer backbone ────────────────────────────
        self.roberta = RobertaModel.from_pretrained(config.MODEL_NAME)

        if config.FREEZE_ROBERTA:
            for param in self.roberta.parameters():
                param.requires_grad = False

        hidden_size = self.roberta.config.hidden_size  # 768

        # ── MLP head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.NUM_CORAL_OUTPUTS),  # 4 cumulative logits
            nn.Sigmoid(),
        )

    # ── helpers for optimizer param-groups ───────────────────
    def roberta_parameters(self):
        """Parameters of the RoBERTa backbone."""
        return self.roberta.parameters()

    def head_parameters(self):
        """Parameters of the MLP head."""
        return self.head.parameters()

    # ── forward ─────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Parameters
        ----------
        input_ids      : (B, MAX_LEN)
        attention_mask  : (B, MAX_LEN)

        Returns
        -------
        logits : (B, 4)  – 4 sigmoid outputs (cumulative probabilities)
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        logits = self.head(cls_hidden)
        return logits

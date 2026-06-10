"""
model.py – DeBERTa + MLP head for 1-5 skill scoring.

Head type (CORAL vs classifier) and how many top backbone layers to fine-tune are
selected by the active experiment (config.HEAD_TYPE / config.UNFREEZE_LAST_N).
"""

import torch
import torch.nn as nn
from transformers import AutoModel

import config
import heads


def _set_trainable_layers(backbone, unfreeze_last_n: int) -> int:
    """Freeze the whole backbone, then unfreeze its top `unfreeze_last_n` layers.

    Embeddings and all lower transformer blocks stay frozen — the recommended
    recipe for a small dataset. Returns the number of layers actually unfrozen.
    """
    for p in backbone.parameters():
        p.requires_grad = False

    if not unfreeze_last_n:
        return 0

    # DeBERTa-v2/v3 (and BERT/RoBERTa) expose the transformer blocks here.
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise AttributeError(
            f"Could not locate transformer layers on {type(backbone).__name__} "
            f"(expected .encoder.layer) to unfreeze."
        )

    n = min(unfreeze_last_n, len(layers))
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True
    return n


class ScoringModel(nn.Module):
    """
    Architecture
    ────────────
    1. DeBERTa-v3-base  →  [CLS] embedding  (768-d)
    2. MLP head:
         Linear(768 → 256) + ReLU + Dropout
         Linear(256 →  64) + ReLU + Dropout
         Linear( 64 →   out) [+ Sigmoid for CORAL]
       out = 4 cumulative logits (coral) or 5 class logits (classifier).
    """

    def __init__(self):
        super().__init__()

        # ── Transformer backbone ────────────────────────────
        # Force fp32 master weights. transformers 5.x keeps the checkpoint dtype
        # by default (deberta-v3-base ships fp16), and DeBERTa fine-tuning in fp16
        # overflows -> NaN. With fp32 params, bf16 autocast (USE_AMP) gives the
        # speed safely.
        self.backbone = AutoModel.from_pretrained(config.MODEL_NAME).float()
        self.n_unfrozen = _set_trainable_layers(self.backbone, config.UNFREEZE_LAST_N)

        hidden_size = self.backbone.config.hidden_size  # 768

        # ── MLP head (raw logits; coral_loss / cross-entropy handle the rest) ──
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, heads.output_dim(config.HEAD_TYPE)),
        )

    # ── helpers for optimizer param-groups ───────────────────
    def backbone_parameters(self):
        """Parameters of the transformer backbone."""
        return self.backbone.parameters()

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
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        logits = self.head(cls_hidden)
        return logits

"""
heads.py – Head-type-specific output shape, loss, and decoding.

Keeps the CORAL vs classifier difference in one place so model.py, train.py and
evaluate.py stay head-agnostic. Adding a head later means editing only this file.

  "coral"      : K-1 cumulative sigmoids, coral_loss, threshold decode
  "classifier" : K logits, cross-entropy, argmax decode
"""

import torch
import torch.nn as nn
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch

import config

_ce = nn.CrossEntropyLoss()


def output_dim(head_type: str) -> int:
    """Number of output units the MLP head should produce."""
    return config.NUM_CORAL_OUTPUTS if head_type == "coral" else config.NUM_CLASSES


def uses_sigmoid(head_type: str) -> bool:
    """CORAL needs a final sigmoid (cumulative probabilities); classifier emits raw logits."""
    return head_type == "coral"


def compute_loss(head_type: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Loss for a batch. `labels` are 0..K-1 integer levels."""
    if head_type == "coral":
        # model applies sigmoid; invert to raw logits for coral_loss.
        raw_logits = torch.log(logits / (1.0 - logits + 1e-8))
        levels = levels_from_labelbatch(labels, num_classes=config.NUM_CLASSES).to(logits.device)
        return coral_loss(raw_logits, levels)
    # classifier: raw logits + cross-entropy
    return _ce(logits, labels)


def decode(head_type: str, logits: torch.Tensor) -> torch.Tensor:
    """Convert head outputs -> predicted class (0..K-1)."""
    if head_type == "coral":
        return (logits > 0.5).sum(dim=1)
    return logits.argmax(dim=1)

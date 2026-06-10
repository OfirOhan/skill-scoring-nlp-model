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


def compute_loss(head_type: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Loss for a batch. `labels` are 0..K-1 integer levels.

    The head emits RAW logits for both types (no sigmoid). coral_loss applies its
    own logsigmoid internally, so feeding it raw logits is both correct and stable
    — the old sigmoid-then-invert round-trip produced NaNs once fine-tuning pushed
    the logits to saturation.
    """
    if head_type == "coral":
        levels = levels_from_labelbatch(labels, num_classes=config.NUM_CLASSES).to(logits.device)
        return coral_loss(logits.float(), levels)
    # classifier: raw logits + cross-entropy
    return _ce(logits, labels)


def decode(head_type: str, logits: torch.Tensor) -> torch.Tensor:
    """Convert head outputs -> predicted class (0..K-1)."""
    if head_type == "coral":
        return (logits > 0).sum(dim=1)   # raw cumulative logit > 0  <=>  P > 0.5
    return logits.argmax(dim=1)

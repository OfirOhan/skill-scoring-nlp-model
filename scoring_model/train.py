"""
train.py – Training loop with CORAL loss, differential LR, and best-model
checkpointing based on validation MAE.
"""

import torch
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch

import config
from dataset import load_data
from model import ScoringModel


def _predictions_from_coral(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORAL sigmoid outputs → predicted class (0-4)."""
    return (logits > 0.5).sum(dim=1)


def _mae(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean Absolute Error (on 0-4 scale)."""
    return (preds - labels).abs().float().mean().item()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── data ────────────────────────────────────────────────
    train_loader, val_loader, _ = load_data()

    # ── model ───────────────────────────────────────────────
    model = ScoringModel().to(device)

    # ── optimizer (two param-groups) ────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.roberta_parameters(), "lr": config.ROBERTA_LR},
        {"params": model.head_parameters(),    "lr": config.MLP_LR},
    ])

    best_val_mae = float("inf")

    # ── training loop ───────────────────────────────────────
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            # CORAL loss expects integer labels (0 … K-1) and logits
            # before sigmoid – but our model already applies sigmoid,
            # so we invert it to get raw logits for coral_loss.
            raw_logits = torch.log(logits / (1.0 - logits + 1e-8))
            levels = levels_from_labelbatch(labels, num_classes=config.NUM_CLASSES).to(device)
            loss = coral_loss(raw_logits, levels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # ── validation ──────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask  = batch["attention_mask"].to(device)
                labels          = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                preds = _predictions_from_coral(logits)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_mae = _mae(all_preds, all_labels)

        print(f"Epoch {epoch:>2}/{config.EPOCHS}  |  "
              f"train_loss={avg_train_loss:.4f}  |  val_MAE={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"  ↳ Saved best model (val_MAE={val_mae:.4f})")

    print(f"\nTraining complete.  Best val MAE = {best_val_mae:.4f}")


if __name__ == "__main__":
    train()

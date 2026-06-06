"""
evaluate.py – Load best checkpoint, run on the test set, and report
MAE, Accuracy, and Confusion Matrix.
"""

import torch
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

import config
from dataset import load_data
from model import ScoringModel


def _predictions_from_coral(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORAL sigmoid outputs → predicted class (0-4)."""
    return (logits > 0.5).sum(dim=1)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── data (only test set needed) ─────────────────────────
    _, _, test_loader = load_data()

    # ── load best model ─────────────────────────────────────
    model = ScoringModel().to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = _predictions_from_coral(logits)  # 0-4

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ── shift back to original 1-5 scale ────────────────────
    all_preds_orig  = all_preds + 1
    all_labels_orig = all_labels + 1

    # ── metrics ─────────────────────────────────────────────
    mae = mean_absolute_error(all_labels_orig, all_preds_orig)
    acc = accuracy_score(all_labels_orig, all_preds_orig)
    cm  = confusion_matrix(all_labels_orig, all_preds_orig, labels=[1, 2, 3, 4, 5])

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    print(f"  MAE      : {mae:.4f}")
    print(f"  Accuracy : {acc:.4f}  ({int(acc * len(all_labels_orig))}/{len(all_labels_orig)})")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: 1  2  3  4  5")
    for i, row in enumerate(cm):
        print(f"    {i + 1}: {row}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()

"""
train.py – Training loop with per-experiment head/loss, differential LR, and
best-model checkpointing (by validation MAE) into runs/<EXPERIMENT>/.
"""

import argparse
import json
import os

import torch

import config
import heads
import metrics
from dataset import load_data
from model import ScoringModel


def train(subset: float | None = None, epochs: int | None = None):
    """Train the active experiment (config.EXPERIMENT).

    Parameters
    ----------
    subset : fraction (0-1) or row count (>=1) of the data to use. Intended for
             quick smoke runs to confirm the model learns (loss falls, val MAE
             drops) before committing to a full run.
    epochs : override config.EPOCHS (handy for short subset runs).
    """
    epochs = epochs or config.EPOCHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = config.USE_AMP and device.type == "cuda"
    print(f"[{config.EXPERIMENT}] {config.EXPERIMENTS[config.EXPERIMENT]['desc']}")
    print(f"Using device: {device}  |  head={config.HEAD_TYPE}  |  "
          f"unfreeze_last_n={config.UNFREEZE_LAST_N}  |  epochs={epochs}  |  "
          f"amp={amp_enabled}{f'  |  subset={subset}' if subset else ''}")

    # ── data ────────────────────────────────────────────────
    train_loader, val_loader, _ = load_data(subset=subset)

    # ── model ───────────────────────────────────────────────
    model = ScoringModel().to(device)

    # ── optimizer ───────────────────────────────────────────
    # Head always trains; the unfrozen top backbone layers (if any) train at the
    # lower BACKBONE_LR. Frozen params are excluded from the optimizer entirely.
    param_groups = [{"params": model.head_parameters(), "lr": config.HEAD_LR}]
    trainable_backbone = [p for p in model.backbone_parameters() if p.requires_grad]
    if trainable_backbone:
        param_groups.append({"params": trainable_backbone, "lr": config.BACKBONE_LR})
        print(f"Fine-tuning top {model.n_unfrozen} backbone layer(s).")
    optimizer = torch.optim.AdamW(param_groups)

    best_val_mae = float("inf")
    epochs_no_improve = 0
    patience = config.EARLY_STOP_PATIENCE
    ckpt = config.checkpoint_path()
    history = []   # per-epoch metrics for the learning-curve graph

    # ── training loop ───────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["label"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(input_ids, attention_mask)
                loss = heads.compute_loss(config.HEAD_TYPE, logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
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

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    logits = model(input_ids, attention_mask)
                preds = heads.decode(config.HEAD_TYPE, logits)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds  = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        val = metrics.ordinal_metrics(all_labels, all_preds)
        val_mae = val["mae"]

        print(f"Epoch {epoch:>2}/{epochs}  |  "
              f"train_loss={avg_train_loss:.4f}  |  "
              f"val_MAE={val_mae:.4f}  |  val_QWK={val['qwk']:.4f}  |  "
              f"val_+-1={val['off_by_one']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_mae": val["mae"],
            "val_qwk": val["qwk"],
            "val_off_by_one": val["off_by_one"],
        })

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Saved best model (val_MAE={val_mae:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stop: no val_MAE improvement in {patience} epochs.")
                break

    hist_path = os.path.join(config.run_dir(), "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.  Best val MAE = {best_val_mae:.4f}")
    print(f"Checkpoint: {ckpt}")
    print(f"History:    {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a skill-scoring experiment")
    parser.add_argument(
        "--experiment", choices=list(config.EXPERIMENTS), default=None,
        help=f"Which variant to train (default: config.EXPERIMENT = {config.EXPERIMENT}).",
    )
    parser.add_argument(
        "--subset", type=float, default=None,
        help="Train on a subset to sanity-check learning before a full run. "
             "Fraction in (0,1) e.g. 0.05, or an absolute row count e.g. 200.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override config.EPOCHS (useful for quick subset runs).",
    )
    args = parser.parse_args()
    if args.experiment:
        config.apply_experiment(args.experiment)
    train(subset=args.subset, epochs=args.epochs)

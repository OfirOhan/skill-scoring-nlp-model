"""
evaluate.py – Load best checkpoint, run on the test set, and report a full
diagnostic: ordinal scoring metrics, retrieval metrics, and — most usefully —
how the two interact (does the scorer fail because the retriever fed it the
wrong chunks, or in spite of good retrieval?).

Sections
────────
1. Scoring   : MAE, exact accuracy, ±1 accuracy, Quadratic Weighted Kappa,
               Spearman ρ, per-class accuracy, confusion matrix.
2. Retrieval : Hit@k, Precision@k, MRR over the test rows, plus a breakdown by
               true skill level (where does retrieval struggle?).
3. Joint     : scoring error conditioned on whether retrieval actually hit an
               evidence document — isolates retrieval-caused scoring errors.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import config
import heads
import metrics
from dataset import load_data_with_frames
from model import ScoringModel


def _load_retrieval_meta() -> dict:
    """Load retrieval_meta.jsonl into {row_id: meta}. Empty if not built yet."""
    path = config.RETRIEVAL_META_PATH
    if not os.path.exists(path):
        return {}
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                meta[obj["row_id"]] = obj
    return meta


def _report_scoring(labels_orig, preds_orig):
    m = metrics.ordinal_metrics(labels_orig, preds_orig)

    print("\n" + "=" * 56)
    print("1. SCORING (ordinal, 1-5)")
    print("=" * 56)
    print(f"  MAE            : {m['mae']:.4f}")
    print(f"  Exact accuracy : {m['accuracy']:.4f}")
    print(f"  +/-1 accuracy  : {m['off_by_one']:.4f}")
    print(f"  Quadratic Weighted Kappa : {m['qwk']:.4f}")
    print(f"  Spearman rho   : {m['spearman']:.4f}")

    cm = confusion_matrix(labels_orig, preds_orig, labels=[1, 2, 3, 4, 5])
    print("\n  Per-class accuracy:")
    for i, label in enumerate([1, 2, 3, 4, 5]):
        support = cm[i].sum()
        correct = cm[i][i]
        acc = (correct / support) if support else 0.0
        print(f"    {label}: {acc:.3f}  ({correct}/{support})")

    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print("  Labels:  1   2   3   4   5")
    for i, row in enumerate(cm):
        print(f"    {i + 1}: " + " ".join(f"{v:3d}" for v in row))
    return m


def _report_retrieval(test_df, retrieval_meta):
    """Grade retrieval on exactly the test rows, overall and by skill level."""
    row_ids = [int(idx) for idx in test_df.index]
    grades = []
    grades_by_level = {lvl: [] for lvl in range(1, 6)}

    for rid in row_ids:
        meta = retrieval_meta.get(rid)
        if meta is None:
            continue
        grade = {
            "hit": meta["hit"],
            "precision": meta["precision"],
            "rr": meta["rr"],
            "evaluable": meta["evaluable"],
        }
        grades.append(grade)
        grades_by_level[int(meta["label"])].append(grade)

    print("\n" + "=" * 56)
    print("2. RETRIEVAL (vs. evidence ground truth, test rows)")
    print("=" * 56)
    if not grades:
        print("  No retrieval metadata found — run build_dataset.py first.")
        return None, None

    agg = metrics.aggregate_retrieval(grades)
    print(f"  Hit@{config.RETRIEVE_TOP_K}       : {agg['hit_rate']:.4f}")
    print(f"  Precision@{config.RETRIEVE_TOP_K} : {agg['precision']:.4f}")
    print(f"  MRR            : {agg['mrr']:.4f}")
    print(f"  Evaluated rows : {agg['n']}  (skipped {agg['n_skipped']} with no evidence)")

    print("\n  By true skill level:")
    print("    lvl   hit   prec    mrr     n")
    for lvl in range(1, 6):
        a = metrics.aggregate_retrieval(grades_by_level[lvl])
        print(f"    {lvl}    {a['hit_rate']:.3f}  {a['precision']:.3f}  {a['mrr']:.3f}  {a['n']:4d}")
    return row_ids, agg


def _report_joint(test_df, preds_orig, labels_orig, retrieval_meta, row_ids):
    """Scoring error conditioned on retrieval success.

    If 'retrieval hit' rows score much better than 'retrieval miss' rows, the
    retriever is a bottleneck on the scorer — the single most actionable signal
    this evaluation produces.
    """
    print("\n" + "=" * 56)
    print("3. JOINT (scoring error vs. retrieval success)")
    print("=" * 56)
    if not retrieval_meta:
        print("  No retrieval metadata found — run build_dataset.py first.")
        return

    preds_orig = np.asarray(preds_orig)
    labels_orig = np.asarray(labels_orig)

    hit_mask, miss_mask = [], []
    for i, rid in enumerate(row_ids):
        meta = retrieval_meta.get(rid)
        if meta is None or not meta["evaluable"]:
            continue
        (hit_mask if meta["hit"] >= 1.0 else miss_mask).append(i)

    def _summary(name, idxs):
        if not idxs:
            print(f"  {name:18s}: (no rows)")
            return
        idxs = np.array(idxs)
        mae = np.abs(preds_orig[idxs] - labels_orig[idxs]).mean()
        acc = (preds_orig[idxs] == labels_orig[idxs]).mean()
        print(f"  {name:18s}: MAE={mae:.4f}  exact_acc={acc:.4f}  n={len(idxs)}")

    _summary("retrieval HIT", hit_mask)
    _summary("retrieval MISS", miss_mask)
    print("\n  (Large gap => retrieval is limiting the scorer; close gap => "
          "scoring errors are intrinsic.)")


def evaluate() -> dict:
    """Evaluate the active experiment's checkpoint; print + persist metrics.json."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = config.USE_AMP and device.type == "cuda"
    print(f"[{config.EXPERIMENT}] {config.EXPERIMENTS[config.EXPERIMENT]['desc']}")
    print(f"Using device: {device}  |  head={config.HEAD_TYPE}  |  amp={amp_enabled}")

    ckpt = config.checkpoint_path()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint for '{config.EXPERIMENT}' at {ckpt} — train it first.")

    # ── data (test set + test frame for retrieval join) ─────
    _, _, test_loader, test_df = load_data_with_frames()

    # ── load best model ─────────────────────────────────────
    model = ScoringModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["label"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(input_ids, attention_mask)
            preds = heads.decode(config.HEAD_TYPE, logits)  # 0-4

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ── shift back to original 1-5 scale ────────────────────
    preds_orig  = all_preds + 1
    labels_orig = all_labels + 1

    retrieval_meta = _load_retrieval_meta()

    scoring = _report_scoring(labels_orig, preds_orig)
    row_ids, retr_agg = _report_retrieval(test_df, retrieval_meta)
    if row_ids is not None:
        _report_joint(test_df, preds_orig, labels_orig, retrieval_meta, row_ids)
    print("=" * 56)

    # ── persist metrics for cross-experiment comparison ─────
    cm = confusion_matrix(labels_orig, preds_orig, labels=[1, 2, 3, 4, 5])
    per_class = {
        int(lbl): (float(cm[i][i] / cm[i].sum()) if cm[i].sum() else 0.0)
        for i, lbl in enumerate([1, 2, 3, 4, 5])
    }
    result = {
        "experiment": config.EXPERIMENT,
        "desc": config.EXPERIMENTS[config.EXPERIMENT]["desc"],
        "model_name": config.MODEL_NAME,
        "head_type": config.HEAD_TYPE,
        "unfreeze_last_n": config.UNFREEZE_LAST_N,
        "n_test": int(len(labels_orig)),
        "scoring": scoring,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "retrieval": retr_agg,
    }
    out = config.metrics_path()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics -> {out}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a skill-scoring experiment")
    parser.add_argument(
        "--experiment", choices=list(config.EXPERIMENTS), default=None,
        help=f"Which variant to evaluate (default: config.EXPERIMENT = {config.EXPERIMENT}).",
    )
    args = parser.parse_args()
    if args.experiment:
        config.apply_experiment(args.experiment)
    evaluate()

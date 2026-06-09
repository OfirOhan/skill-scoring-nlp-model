"""
metrics.py – Shared metric helpers for the scoring model.

Two families:
  • Ordinal metrics  – grade the 1-5 score predictions (MAE, exact/±1 accuracy,
                       Quadratic Weighted Kappa, Spearman ρ).
  • Retrieval metrics – grade whether the RAG surfaced the chunks that actually
                       carry evidence for the queried skill (Hit@k, Precision@k,
                       Reciprocal Rank).

Keeping both here lets train.py, evaluate.py, and build_dataset.py report the
same numbers without copy-paste drift.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    cohen_kappa_score,
)


# ──────────────────────────────────────────────────────────────
# Ordinal (scoring) metrics
# ──────────────────────────────────────────────────────────────
def ordinal_metrics(y_true, y_pred) -> dict:
    """Compute ordinal-regression metrics on integer labels (any consistent scale).

    Returns
    -------
    dict with keys: mae, accuracy, off_by_one, qwk, spearman
        off_by_one : fraction of predictions within ±1 of the true label
        qwk        : Quadratic Weighted Kappa — the canonical ordinal agreement
                     metric; penalises predictions more the further they miss.
        spearman   : rank correlation between predicted and true scores
                     (NaN-safe: 0.0 when a prediction vector has no variance).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    off_by_one = float(np.mean(np.abs(y_true - y_pred) <= 1))

    # QWK needs a fixed label set so the weighting is stable across splits.
    labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels))

    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        rho = 0.0
    else:
        rho = float(spearmanr(y_true, y_pred).correlation)

    return {
        "mae": mae,
        "accuracy": acc,
        "off_by_one": off_by_one,
        "qwk": qwk,
        "spearman": rho,
    }


# ──────────────────────────────────────────────────────────────
# Retrieval metrics
# ──────────────────────────────────────────────────────────────
def retrieval_row_metrics(retrieved_doc_ids, relevant_doc_ids) -> dict:
    """Grade a single retrieval against its ground-truth relevant documents.

    Parameters
    ----------
    retrieved_doc_ids : ordered list of doc_ids the RAG returned (best first).
                        May contain None for chunks whose source could not be
                        mapped — these never count as a hit.
    relevant_doc_ids  : set/list of doc_ids that actually carry evidence for
                        the queried skill.

    Returns
    -------
    dict with keys: hit, precision, rr, num_relevant, evaluable
        hit       : 1.0 if any retrieved doc is relevant, else 0.0
        precision : fraction of retrieved chunks from relevant docs (Precision@k)
        rr        : reciprocal rank of the first relevant doc (0.0 if none)
        evaluable : False when there are no relevant docs (skill not evidenced
                    in any document) — such rows are excluded from averages.
    """
    relevant = set(d for d in relevant_doc_ids if d is not None)
    retrieved = list(retrieved_doc_ids)

    if not relevant:
        return {
            "hit": 0.0,
            "precision": 0.0,
            "rr": 0.0,
            "num_relevant": 0,
            "evaluable": False,
        }

    hits = [1 if d in relevant else 0 for d in retrieved]
    hit = 1.0 if any(hits) else 0.0
    precision = (sum(hits) / len(retrieved)) if retrieved else 0.0

    rr = 0.0
    for rank, is_hit in enumerate(hits, start=1):
        if is_hit:
            rr = 1.0 / rank
            break

    return {
        "hit": hit,
        "precision": precision,
        "rr": rr,
        "num_relevant": len(relevant),
        "evaluable": True,
    }


def aggregate_retrieval(rows: list[dict]) -> dict:
    """Average per-row retrieval metrics, ignoring non-evaluable rows.

    `rows` is a list of dicts as returned by retrieval_row_metrics.
    """
    evaluable = [r for r in rows if r.get("evaluable")]
    n = len(evaluable)
    if n == 0:
        return {"hit_rate": 0.0, "precision": 0.0, "mrr": 0.0, "n": 0, "n_skipped": len(rows)}

    return {
        "hit_rate": sum(r["hit"] for r in evaluable) / n,
        "precision": sum(r["precision"] for r in evaluable) / n,
        "mrr": sum(r["rr"] for r in evaluable) / n,
        "n": n,
        "n_skipped": len(rows) - n,
    }

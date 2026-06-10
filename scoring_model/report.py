"""
report.py – Consolidated evaluation across ALL experiments.

For every experiment that has a trained checkpoint, this:
  1. runs the model on the held-out TEST set (via evaluate.evaluate, which saves
     a rich runs/<exp>/metrics.json: scoring + per-class + confusion + retrieval);
  2. plots the LEARNING CURVE from runs/<exp>/history.json
     (train_loss / val MAE / val QWK per epoch) -> runs/<exp>/learning_curve.png;
  3. saves a combined comparison figure + a Markdown REPORT (runs/report.md).

    python scoring_model/report.py

Needs matplotlib (headless 'Agg' backend, fine on the pod).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")            # no display needed
import matplotlib.pyplot as plt

import config
import evaluate as evaluate_mod


def _load_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def _plot_learning_curve(exp, history, out_path):
    """Per-model: train_loss + val MAE (left) and val QWK (right) vs epoch."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], "-o", label="train_loss")
    ax1.plot(epochs, [h["val_mae"] for h in history], "-s", label="val_MAE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss / MAE"); ax1.set_title(f"{exp} — loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["val_qwk"] for h in history], "-^", color="tab:green", label="val_QWK")
    ax2.plot(epochs, [h["val_off_by_one"] for h in history], "-d", color="tab:purple", label="val_±1")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.set_title(f"{exp} — val metrics")
    ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(histories, results, out_path):
    """Two panels: val-QWK learning curves (all models) + test-metric bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for exp, hist in histories.items():
        if hist:
            ax1.plot([h["epoch"] for h in hist], [h["val_qwk"] for h in hist], "-o", label=exp)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("val QWK"); ax1.set_title("Learning (val QWK)")
    ax1.set_ylim(-0.05, 1.05); ax1.legend(); ax1.grid(alpha=0.3)

    exps = list(results.keys())
    x = range(len(exps))
    mae = [results[e]["scoring"]["mae"] for e in exps]
    qwk = [results[e]["scoring"]["qwk"] for e in exps]
    w = 0.35
    ax2.bar([i - w / 2 for i in x], qwk, w, label="test QWK", color="tab:green")
    ax2.bar([i + w / 2 for i in x], mae, w, label="test MAE", color="tab:red")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(exps, rotation=15, ha="right")
    ax2.set_title("Test metrics"); ax2.legend(); ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _confusion_block(cm) -> str:
    lines = ["          pred", "       " + "".join(f"{l:>5}" for l in [1, 2, 3, 4, 5])]
    for i, row in enumerate(cm):
        lines.append(f"  true {i + 1}:" + "".join(f"{v:>5}" for v in row))
    return "```\n" + "\n".join(lines) + "\n```"


def build():
    runs_dir = config._RUNS_DIR
    results, histories = {}, {}

    for exp in config.EXPERIMENTS:
        config.apply_experiment(exp)
        if not os.path.exists(config.checkpoint_path()):
            print(f"[report] skipping {exp} — no checkpoint")
            continue

        print(f"\n[report] evaluating {exp} on test set ...")
        evaluate_mod.evaluate()                       # runs test, saves metrics.json

        results[exp] = _load_json(config.metrics_path())
        histories[exp] = _load_json(os.path.join(config.run_dir(), "history.json"))
        if histories[exp]:
            _plot_learning_curve(exp, histories[exp],
                                 os.path.join(config.run_dir(), "learning_curve.png"))

    if not results:
        print("No evaluated models found. Train them first (run_all.py).")
        return

    combined_png = os.path.join(str(runs_dir), "report_combined.png")
    _plot_combined(histories, results, combined_png)

    # ── Markdown report ─────────────────────────────────────
    order = sorted(results, key=lambda e: results[e]["scoring"]["mae"])
    lines = ["# Skill-Scoring — Model Comparison Report", ""]
    lines += ["## Test-set metrics (sorted by MAE)", "",
              "| experiment | MAE | acc | ±1 | QWK | Spearman | n |",
              "|---|---|---|---|---|---|---|"]
    for e in order:
        s = results[e]["scoring"]
        lines.append(f"| {e} | {s['mae']:.4f} | {s['accuracy']:.4f} | {s['off_by_one']:.4f} "
                     f"| {s['qwk']:.4f} | {s['spearman']:.4f} | {results[e]['n_test']} |")
    lines += ["", f"**Best by MAE:** `{order[0]}`  |  **Best by QWK:** "
                  f"`{max(results, key=lambda e: results[e]['scoring']['qwk'])}`", ""]
    lines += ["## Learning curves & test metrics", "", f"![combined](report_combined.png)", ""]

    for e in order:
        r = results[e]
        lines += [f"## {e}", f"_{r['desc']}_", ""]
        if histories.get(e):
            lines += [f"![learning curve]({e}/learning_curve.png)", ""]
        pc = r.get("per_class_accuracy", {})
        lines += ["Per-class accuracy: " + ", ".join(f"L{k}={v:.2f}" for k, v in pc.items()), ""]
        if r.get("confusion_matrix"):
            lines += ["Confusion matrix:", _confusion_block(r["confusion_matrix"]), ""]
        if r.get("retrieval"):
            rt = r["retrieval"]
            lines += [f"Retrieval: Hit={rt.get('hit_rate', 0):.3f}  "
                      f"Prec={rt.get('precision', 0):.3f}  MRR={rt.get('mrr', 0):.3f}", ""]

    report_path = os.path.join(str(runs_dir), "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[report] wrote {report_path}")
    print(f"[report] combined figure: {combined_png}")
    print(f"[report] per-model curves: runs/<exp>/learning_curve.png")


if __name__ == "__main__":
    build()

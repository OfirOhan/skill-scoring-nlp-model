"""
run_all.py – Train + evaluate every experiment, then print the comparison table.

    python scoring_model/run_all.py --smoke     # fast end-to-end sanity check
    python scoring_model/run_all.py             # full runs, all 3 variants
    python scoring_model/run_all.py --experiments coral_frozen coral_top3

--smoke trains on a tiny stratified subset for a couple of epochs so you can
confirm the whole pipeline (data -> train -> save -> evaluate -> compare) works
before committing GPU hours to full runs.
"""

import argparse

import config
import report as report_mod
from train import train
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Run and compare all scoring experiments")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast sanity run: small subset + few epochs (default subset=0.05, epochs=2).")
    parser.add_argument("--subset", type=float, default=None,
                        help="Train-data fraction (0-1) or row count (>=1). Overrides --smoke default.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs per experiment. Overrides --smoke default.")
    parser.add_argument("--experiments", nargs="+", choices=list(config.EXPERIMENTS),
                        default=list(config.EXPERIMENTS),
                        help="Which experiments to run (default: all).")
    args = parser.parse_args()

    if args.smoke:
        # Small subset, but enough epochs to SEE the trend: a correctly-wired model
        # should overfit ~100 rows -> train_loss falls steadily and metrics climb.
        subset = args.subset if args.subset is not None else 0.05
        epochs = args.epochs if args.epochs is not None else 10
    else:
        subset, epochs = args.subset, args.epochs

    mode = "SMOKE" if args.smoke else "FULL"
    print(f"\n########## RUN ALL ({mode}) — {len(args.experiments)} experiment(s) "
          f"| subset={subset} epochs={epochs or config.EPOCHS} ##########")

    for exp in args.experiments:
        config.apply_experiment(exp)
        print(f"\n========================= {exp} =========================")
        train(subset=subset, epochs=epochs)

    # Consolidated report: test-set eval + learning curves + saved report.md
    print("\n########## REPORT ##########")
    report_mod.build()


if __name__ == "__main__":
    main()

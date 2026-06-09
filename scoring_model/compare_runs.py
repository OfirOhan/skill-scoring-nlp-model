"""
compare_runs.py – Side-by-side comparison of saved experiment runs.

Reads every runs/<experiment>/metrics.json (written by evaluate.py) and prints
one sortable table, so the 3 variants can be compared at a glance.

    python scoring_model/compare_runs.py
"""

import glob
import json
import os

import config


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "  -  "


def main():
    pattern = os.path.join(str(config._RUNS_DIR), "*", "metrics.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No runs found under {config._RUNS_DIR}.")
        print("Train + evaluate at least one experiment first.")
        return

    rows = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        s = d.get("scoring", {})
        rows.append({
            "experiment": d.get("experiment", os.path.basename(os.path.dirname(f))),
            "mae": s.get("mae"),
            "acc": s.get("accuracy"),
            "pm1": s.get("off_by_one"),
            "qwk": s.get("qwk"),
            "spearman": s.get("spearman"),
            "n": d.get("n_test"),
        })

    # Best ordinal agreement first: lowest MAE.
    rows.sort(key=lambda r: (r["mae"] is None, r["mae"]))

    hdr = (f'{"experiment":20s} {"MAE":>7} {"acc":>7} {"+-1":>7} '
           f'{"QWK":>7} {"spearman":>9} {"n":>6}')
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f'{r["experiment"]:20s} {_fmt(r["mae"]):>7} {_fmt(r["acc"]):>7} '
              f'{_fmt(r["pm1"]):>7} {_fmt(r["qwk"]):>7} {_fmt(r["spearman"]):>9} '
              f'{str(r["n"]):>6}')

    print(f"\nBest by MAE: {rows[0]['experiment']}   "
          f"(QWK leader: {max(rows, key=lambda r: r['qwk'] or -1)['experiment']})")


if __name__ == "__main__":
    main()

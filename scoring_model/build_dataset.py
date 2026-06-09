"""
build_dataset.py – The missing bridge: RAG  →  scoring-model training data.

For every (persona, skill) pair this script:

    1. ingests the persona's synthetic documents into a per-persona ChromaDB
       collection (once, idempotently);
    2. runs the RAG retriever with the *skill name* as the query;
    3. writes the top-k chunks as (chunk1, chunk2, chunk3) and the persona's
       declared skill level (1-5) as the label.

It produces two aligned artifacts (same row order):

    data/training_data.csv     skill, chunk1, chunk2, chunk3, label
                               → consumed directly by train.py / evaluate.py

    data/retrieval_meta.jsonl  one JSON object per row carrying the retrieved
                               doc_ids, the ground-truth evidence doc_ids, and
                               per-row retrieval metrics (hit / precision / rr)
                               → consumed by evaluate.py to grade the retriever

The retrieval ground truth comes for free from generation: each document's
`skill_evidence[skill]` intensity (1-5) records how strongly that document was
written to demonstrate the skill. A document is a *relevant* retrieval target
when that intensity is high enough (see `_relevant_doc_ids`).

Run from the project root:
    python scoring_model/build_dataset.py                 # full corpus (fast: no LLM expansion)
    python scoring_model/build_dataset.py --limit 20      # quick smoke run
    python scoring_model/build_dataset.py --expand        # add LLM query expansion (slower, wider recall)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --- make both project-root packages (rag) and sibling modules importable ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:                       # works when run as a script (scoring_model/ on path)
    import config
    import metrics
except ModuleNotFoundError:  # works when run as `python -m scoring_model.build_dataset`
    from scoring_model import config, metrics

from rag.ingest import ingest_text, get_collection
from rag.retriever import retrieve_batch_for_training


# ──────────────────────────────────────────────────────────────
# Evidence ground truth
# ──────────────────────────────────────────────────────────────
def _evidence_intensity(doc: dict, skill: str) -> int:
    """How strongly `doc` demonstrates `skill` (0 if not mentioned).

    skill_evidence carries both display ('AWS') and normalized ('aws') keys;
    persona skills use the display form, so we try that first and fall back to
    a normalized lookup for robustness.
    """
    evidence = doc.get("skill_evidence", {})
    if skill in evidence:
        return int(evidence[skill])
    norm = skill.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return int(evidence.get(norm, 0))


def _relevant_doc_ids(skill: str, level: int, persona_docs: list[dict]) -> set[str]:
    """The set of doc_ids that genuinely carry evidence for `skill`.

    • level ≥ 2 : docs whose intensity meets EVIDENCE_THRESHOLD. If the LLM
      allocator never marked any doc that high (rare), fall back to the
      strongest doc(s) so the row stays evaluable.
    • level == 1 : the skill is only ever a passing mention, so any document
      that references it at all is the best evidence available.
    """
    intensities = {d["doc_id"]: _evidence_intensity(d, skill) for d in persona_docs}
    mentioned = {d: i for d, i in intensities.items() if i >= 1}
    if not mentioned:
        return set()

    if level >= 2:
        relevant = {d for d, i in mentioned.items() if i >= config.EVIDENCE_THRESHOLD}
        if not relevant:
            top = max(mentioned.values())
            relevant = {d for d, i in mentioned.items() if i == top}
        return relevant

    return set(mentioned)  # level 1


# ──────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────
def build(limit: int | None = None, top_k: int | None = None, expand: bool = False):
    top_k = top_k or config.RETRIEVE_TOP_K

    with open(config.PERSONAS_PATH, encoding="utf-8") as f:
        personas = json.load(f)
    with open(config.DOCUMENTS_DB_PATH, encoding="utf-8") as f:
        documents_db = json.load(f)

    docs_by_persona: dict[str, list[dict]] = defaultdict(list)
    for doc in documents_db.values():
        docs_by_persona[doc["persona_id"]].append(doc)

    if limit is not None:
        personas = personas[:limit]

    print(f"Building training data from {len(personas)} personas "
          f"(top_k={top_k}, expand={expand})")

    csv_rows: list[dict] = []
    meta_rows: list[dict] = []
    retrieval_grades: list[dict] = []
    row_id = 0

    for p_idx, persona in enumerate(personas):
        pid = persona["persona_id"]
        persona_docs = docs_by_persona.get(pid, [])
        if not persona_docs:
            continue

        # --- Ingest once (idempotent via persistent ChromaDB) ---
        collection = get_collection(pid)
        if collection.count() == 0:
            for doc in persona_docs:
                ingest_text(doc.get("text", ""), pid, doc["doc_id"], doc.get("type", "cv"))

        # --- Retrieve all skills in one batch (shared corpus + BM25 index) ---
        skill_items = list(persona["skills"].items())
        skill_names = [s for s, _ in skill_items]
        batch = retrieve_batch_for_training(skill_names, pid, top_k=top_k, expand=expand)

        # --- One training row per skill ---
        for (skill, level), res in zip(skill_items, batch):
            chunks = res["chunks"]
            retrieved_doc_ids = res["doc_ids"]

            padded = (chunks + ["", "", ""])[:3]
            csv_rows.append({
                "skill": skill,
                "chunk1": padded[0],
                "chunk2": padded[1],
                "chunk3": padded[2],
                "label": int(level),
            })

            relevant = _relevant_doc_ids(skill, level, persona_docs)
            grade = metrics.retrieval_row_metrics(retrieved_doc_ids, relevant)
            retrieval_grades.append(grade)

            meta_rows.append({
                "row_id": row_id,
                "persona_id": pid,
                "skill": skill,
                "label": int(level),
                "retrieved_doc_ids": retrieved_doc_ids,
                "evidence_doc_ids": sorted(relevant),
                "hit": grade["hit"],
                "precision": grade["precision"],
                "rr": grade["rr"],
                "evaluable": grade["evaluable"],
            })
            row_id += 1

        if (p_idx + 1) % 10 == 0:
            print(f"  ...{p_idx + 1}/{len(personas)} personas -> {row_id} rows")

    # --- Write CSV (column order matches the training schema) ---
    Path(config.DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(csv_rows, columns=["skill", "chunk1", "chunk2", "chunk3", "label"])
    df.to_csv(config.DATA_PATH, index=False)

    # --- Write retrieval sidecar (row_id aligns with CSV row order) ---
    with open(config.RETRIEVAL_META_PATH, "w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # --- Summary ---
    agg = metrics.aggregate_retrieval(retrieval_grades)
    print("\n" + "=" * 56)
    print("DATASET BUILD COMPLETE")
    print("=" * 56)
    print(f"  Rows written      : {len(csv_rows)}")
    print(f"  CSV               : {config.DATA_PATH}")
    print(f"  Retrieval meta    : {config.RETRIEVAL_META_PATH}")
    print(f"  Label distribution: "
          f"{df['label'].value_counts().sort_index().to_dict()}")
    print("\n  Retrieval quality (vs. evidence ground truth):")
    print(f"    Hit@{top_k}        : {agg['hit_rate']:.3f}")
    print(f"    Precision@{top_k}  : {agg['precision']:.3f}")
    print(f"    MRR             : {agg['mrr']:.3f}")
    print(f"    Evaluated rows  : {agg['n']}  (skipped {agg['n_skipped']} with no evidence)")
    print("=" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG-backed scoring-model training data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N personas (quick runs)")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"Chunks retrieved per skill (default: {config.RETRIEVE_TOP_K})")
    parser.add_argument("--expand", action="store_true",
                        help="Enable LLM query expansion (wider recall, but an LLM "
                             "call per skill — much slower). Off by default.")
    args = parser.parse_args()

    build(limit=args.limit, top_k=args.top_k, expand=args.expand)

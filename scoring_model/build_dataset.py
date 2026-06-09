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
import os
import sys
from collections import Counter, defaultdict
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


def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


# ──────────────────────────────────────────────────────────────
# Data-quality exclusion rules (1, 3, 4) — see config.* thresholds
# ──────────────────────────────────────────────────────────────
def _load_exclusion_inputs(documents_db: dict):
    """Read the validation reports and precompute the flag sets for rules 1 & 3.

    Returns (flagged_personas, flagged_skills) where:
      flagged_personas : set[persona_id]            (rule 3 — allocation incoherent)
      flagged_skills   : set[(persona_id, norm_skill)] (rule 1 — showcase mismatch)
    """
    doc_persona = {d["doc_id"]: d["persona_id"] for d in documents_db.values()}

    # Rule 3: personas whose evidence allocation was judged incoherent.
    flagged_personas: set[str] = set()
    if os.path.exists(config.ALLOCATION_REPORT_PATH):
        alloc = json.load(open(config.ALLOCATION_REPORT_PATH, encoding="utf-8"))
        flagged_personas = {r["persona_id"] for r in alloc.get("results", []) if r.get("flagged")}

    # Rule 1: (persona, skill) where max|delta|>=MAX or avg|delta|>=AVG across docs.
    flagged_skills: set[tuple[str, str]] = set()
    if os.path.exists(config.SHOWCASE_REPORT_PATH):
        show = json.load(open(config.SHOWCASE_REPORT_PATH, encoding="utf-8"))
        deltas: dict[tuple[str, str], list[int]] = defaultdict(list)
        for r in show.get("results", []):
            pid = doc_persona.get(r["doc_id"])
            if pid is None:
                continue
            for c in r["comparisons"]:
                deltas[(pid, _norm(c["skill"]))].append(abs(c["delta"]))
        for key, ds in deltas.items():
            if max(ds) >= config.SKILL_MISMATCH_MAX or (sum(ds) / len(ds)) >= config.SKILL_MISMATCH_AVG:
                flagged_skills.add(key)

    return flagged_personas, flagged_skills


def _usable_doc(doc: dict) -> bool:
    """Rule 4: skip failed or too-short documents at ingestion."""
    text = doc.get("text", "")
    if "GENERATION FAILED" in text:
        return False
    return len(text.split()) >= config.MIN_DOC_WORDS


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

    flagged_personas, flagged_skills = _load_exclusion_inputs(documents_db)
    print(f"Building training data from {len(personas)} personas "
          f"(top_k={top_k}, expand={expand})")
    print(f"Exclusion inputs: {len(flagged_personas)} flagged personas (rule 3), "
          f"{len(flagged_skills)} flagged (persona,skill) (rule 1)")

    csv_rows: list[dict] = []
    meta_rows: list[dict] = []
    retrieval_grades: list[dict] = []   # only kept (non-excluded) rows
    reason_counts: Counter = Counter()
    row_id = 0

    for p_idx, persona in enumerate(personas):
        pid = persona["persona_id"]
        # Rule 4: drop failed / too-short docs from this persona's corpus.
        persona_docs = [d for d in docs_by_persona.get(pid, []) if _usable_doc(d)]
        if not persona_docs:
            continue

        # Rule 3: skip personas with incoherent evidence allocation entirely.
        if pid in flagged_personas:
            reason_counts["alloc_flagged_persona"] += len(persona["skills"])
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
            padded = (chunks + [""] * top_k)[:top_k]

            # --- Exclusion decision (rules 1 & 2) ---
            exclude_reason = None
            if (pid, _norm(skill)) in flagged_skills:
                exclude_reason = "skill_mismatch"          # rule 1
            elif not padded[0]:
                exclude_reason = "empty_retrieval"         # rule 2
            exclude = exclude_reason is not None
            if exclude:
                reason_counts[exclude_reason] += 1

            row = {"skill": skill}
            for i in range(top_k):
                row[f"chunk{i + 1}"] = padded[i]
            row["label"] = int(level)
            row["exclude"] = int(exclude)
            csv_rows.append(row)

            relevant = _relevant_doc_ids(skill, level, persona_docs)
            grade = metrics.retrieval_row_metrics(retrieved_doc_ids, relevant)
            if not exclude:
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
                "exclude": exclude,
                "exclude_reason": exclude_reason,
            })
            row_id += 1

        if (p_idx + 1) % 10 == 0:
            print(f"  ...{p_idx + 1}/{len(personas)} personas -> {row_id} rows")

    # --- Write CSV (training schema + exclude flag) ---
    Path(config.DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    chunk_cols = [f"chunk{i + 1}" for i in range(top_k)]
    df = pd.DataFrame(csv_rows, columns=["skill", *chunk_cols, "label", "exclude"])
    df.to_csv(config.DATA_PATH, index=False)

    # --- Write retrieval sidecar (row_id aligns with CSV row order) ---
    with open(config.RETRIEVAL_META_PATH, "w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # --- Summary ---
    kept = df[df["exclude"] == 0]
    dropped = int(df["exclude"].sum()) + reason_counts["alloc_flagged_persona"]
    total = len(df) + reason_counts["alloc_flagged_persona"]
    agg = metrics.aggregate_retrieval(retrieval_grades)
    print("\n" + "=" * 56)
    print("DATASET BUILD COMPLETE")
    print("=" * 56)
    print(f"  Rows emitted (CSV): {len(df)}   kept={len(kept)}  excluded={int(df['exclude'].sum())}")
    print(f"  Dropped total     : {dropped}/{total} ({100*dropped/total:.1f}%) incl. flagged personas")
    print("  Drop reasons:")
    for reason, n in reason_counts.most_common():
        print(f"    {reason:24s}: {n}")
    print(f"  CSV               : {config.DATA_PATH}")
    print(f"  Retrieval meta    : {config.RETRIEVAL_META_PATH}")
    print(f"  Kept label dist   : {kept['label'].value_counts().sort_index().to_dict()}")
    print("\n  Retrieval quality (kept rows, vs. evidence ground truth):")
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

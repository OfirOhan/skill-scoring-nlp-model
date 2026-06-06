"""
Phase 6: Assemble checkpoints into final personas.json and documents_db.json.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def assemble_dataset(checkpoint_dir: Path, output_dir: Path = None):
    """Merge all checkpoint files into the final dataset files.

    Reads from:
        checkpoint_dir/personas/       -> persona JSON files
        checkpoint_dir/allocations/    -> allocation JSON files
        checkpoint_dir/documents/      -> document JSON files

    Writes to:
        output_dir/personas.json
        output_dir/documents_db.json
    """
    if output_dir is None:
        output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    personas_dir = checkpoint_dir / "personas"
    docs_dir = checkpoint_dir / "documents"
    alloc_dir = checkpoint_dir / "allocations"

    # --- Assemble personas ---
    personas = []
    if personas_dir.exists():
        for f in sorted(personas_dir.glob("p_*.json")):
            with open(f) as fp:
                persona = json.load(fp)

            # Attach document IDs from doc planning checkpoints
            planning_file = alloc_dir / f"{persona['persona_id']}_docs.json"
            if planning_file.exists():
                with open(planning_file) as fp:
                    doc_plans = json.load(fp)
                persona["document_ids"] = [d["doc_id"] for d in doc_plans]

            personas.append(persona)

    # --- Assemble documents ---
    documents_db = {}
    if docs_dir.exists():
        for f in sorted(docs_dir.glob("d_*.json")):
            with open(f) as fp:
                doc = json.load(fp)
            doc_id = doc["doc_id"]
            documents_db[doc_id] = doc

    # --- Write outputs ---
    personas_file = output_dir / "personas.json"
    with open(personas_file, "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(personas)} personas to {personas_file}")

    docs_file = output_dir / "documents_db.json"
    with open(docs_file, "w", encoding="utf-8") as f:
        json.dump(documents_db, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(documents_db)} documents to {docs_file}")

    # --- Summary stats ---
    total_skills = sum(len(p.get("skills", {})) for p in personas)
    total_docs = len(documents_db)
    doc_types = {}
    for doc in documents_db.values():
        dt = doc.get("type", "unknown")
        doc_types[dt] = doc_types.get(dt, 0) + 1

    logger.info(f"Dataset summary:")
    logger.info(f"  Personas: {len(personas)}")
    logger.info(f"  Total skills: {total_skills}")
    logger.info(f"  Total documents: {total_docs}")
    for dt, count in sorted(doc_types.items()):
        logger.info(f"    {dt}: {count}")

    return personas, documents_db

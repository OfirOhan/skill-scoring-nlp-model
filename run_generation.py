"""
Main orchestration script for SPE synthetic dataset generation.

Usage:
    python run_generation.py --num-personas 50 --concurrency 4
    python run_generation.py --num-personas 300 --concurrency 4 --skip-validation
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import aiohttp
from tqdm.auto import tqdm

from generate.personas import generate_persona
from generate.planner import plan_documents, allocate_evidence
from generate.documents import generate_document
from generate.assembler import assemble_dataset
from generate.seed_generator import generate_all_seeds
from validate.shortcut_check import run_shortcut_checks
from validate.stats import run_stats_checks
from validate.cross_validate import run_validation_checks

# Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
REPORTS_DIR = DATA_DIR / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "generation.log"),
    ],
)
logger = logging.getLogger(__name__)


async def process_persona(
    persona_id: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    structure_seeds: dict | None = None,
) -> dict:
    """Full pipeline for a single persona: generate → plan → allocate → generate docs."""

    # Phase 2: Generate persona
    persona = await generate_persona(
        persona_id, session, semaphore,
        checkpoint_dir=CHECKPOINT_DIR / "personas",
    )

    # Phase 3: Plan documents
    doc_plans = await plan_documents(
        persona, session, semaphore,
        checkpoint_dir=CHECKPOINT_DIR / "allocations",
    )

    # Phase 4: Allocate evidence
    allocation = await allocate_evidence(
        persona, doc_plans, session, semaphore,
        checkpoint_dir=CHECKPOINT_DIR / "allocations",
    )

    # Update persona with document IDs
    persona["document_ids"] = [d["doc_id"] for d in doc_plans]

    # Re-save persona with doc IDs
    persona_file = CHECKPOINT_DIR / "personas" / f"{persona_id}.json"
    with open(persona_file, "w") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)

    # Phase 5: Generate all documents (parallel within persona)
    doc_tasks = [
        generate_document(
            doc_plan, persona, allocation,
            session, semaphore,
            checkpoint_dir=CHECKPOINT_DIR / "documents",
            structure_seeds=structure_seeds,
        )
        for doc_plan in doc_plans
    ]
    documents = await asyncio.gather(*doc_tasks)

    logger.info(
        f"Completed {persona_id}: {persona['name']} — "
        f"{len(persona['skills'])} skills, {len(documents)} docs"
    )
    return persona


async def run_generation(num_personas: int, concurrency: int):
    """Run the full generation pipeline."""
    # Ensure directories exist
    for d in [DATA_DIR, CHECKPOINT_DIR, REPORTS_DIR,
              CHECKPOINT_DIR / "personas",
              CHECKPOINT_DIR / "allocations",
              CHECKPOINT_DIR / "documents"]:
        d.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)

    logger.info(f"Starting generation: {num_personas} personas, concurrency={concurrency}")

    # Pre-step: Generate document structure seeds
    seeds_path = DATA_DIR / "structure_seeds.json"
    structure_seeds = await generate_all_seeds(seeds_path)
    logger.info(f"Loaded {sum(len(v) for v in structure_seeds.values())} structure seeds")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        # Process all personas in parallel (throttled by semaphore), with a
        # progress bar that advances as each persona finishes (any order).
        pbar = tqdm(total=num_personas, desc="Personas", unit="persona")

        async def _tracked(coro):
            try:
                return await coro
            finally:
                pbar.update(1)

        tasks = [
            _tracked(process_persona(f"p_{i:03d}", session, semaphore, structure_seeds=structure_seeds))
            for i in range(num_personas)
        ]
        personas = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()

    # Log any failures
    failures = [i for i, p in enumerate(personas) if isinstance(p, Exception)]
    if failures:
        logger.error(f"{len(failures)} personas failed: {failures}")
        for i in failures:
            logger.error(f"  p_{i:03d}: {personas[i]}")

    successful = [p for p in personas if not isinstance(p, Exception)]
    logger.info(f"Generation complete: {len(successful)}/{num_personas} personas succeeded")

    return successful


async def run_validation(concurrency: int):
    """Run all validation checks on the assembled dataset."""
    logger.info("Running validation...")

    # Load assembled dataset
    personas_file = DATA_DIR / "personas.json"
    docs_file = DATA_DIR / "documents_db.json"

    if not personas_file.exists() or not docs_file.exists():
        logger.error("Dataset files not found — run assembly first")
        return

    with open(personas_file) as f:
        personas = json.load(f)
    with open(docs_file) as f:
        documents_db = json.load(f)

    # Non-LLM checks
    logger.info("Running anti-shortcut checks...")
    run_shortcut_checks(personas, documents_db, REPORTS_DIR)

    logger.info("Running statistical checks...")
    run_stats_checks(personas, documents_db, REPORTS_DIR)

    # LLM checks
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        await run_validation_checks(
            personas, documents_db,
            CHECKPOINT_DIR / "allocations",
            session, semaphore,
            REPORTS_DIR,
        )

    logger.info(f"All validation reports written to {REPORTS_DIR}")


async def main(args):
    if not args.validate_only:
        await run_generation(args.num_personas, args.concurrency)

        # Assemble
        logger.info("Assembling final dataset...")
        assemble_dataset(CHECKPOINT_DIR, DATA_DIR)

    if not args.skip_validation:
        await run_validation(args.concurrency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPE Synthetic Dataset Generation")
    parser.add_argument(
        "--num-personas", type=int, default=50,
        help="Number of personas to generate (default: 50 for pilot)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent LLM calls (default: 4)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip validation checks after generation",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run validation on existing dataset",
    )
    args = parser.parse_args()

    asyncio.run(main(args))

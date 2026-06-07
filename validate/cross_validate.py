"""
Validation: LLM Check A (per-document skill showcase) and Check B (allocation rationality).
All checks are read-only — output reports only, no modification to the DB.
"""

import json
import asyncio
import aiohttp
import logging
from pathlib import Path

from generate.prompts import VALIDATION_SKILL_SHOWCASE, VALIDATION_ALLOCATION_RATIONALITY
from generate.personas import llm_call, _extract_json

logger = logging.getLogger(__name__)


async def check_skill_showcase(
    document: dict,
    skills_to_check: list[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> dict:
    """LLM Check A: Read document blind and judge skill evidence levels.

    Returns dict with comparison results.
    """
    prompt = VALIDATION_SKILL_SHOWCASE.format(
        document_text=document["text"][:3000],  # Cap to avoid huge prompts
        skills_list="\n".join(f"- {s}" for s in skills_to_check),
    )

    try:
        raw = await llm_call(session, semaphore, prompt)
        llm_scores = _extract_json(raw)
    except Exception as e:
        logger.warning(f"Skill showcase check failed for {document['doc_id']}: {e}")
        return {"doc_id": document["doc_id"], "error": str(e), "comparisons": []}

    # Build comparison
    allocated = document.get("skill_evidence", {})
    comparisons = []
    for skill in skills_to_check:
        alloc_level = allocated.get(skill, 1)
        llm_level = llm_scores.get(skill, 1)
        if isinstance(llm_level, str):
            try:
                llm_level = int(llm_level)
            except ValueError:
                llm_level = 1

        delta = abs(alloc_level - llm_level)
        comparisons.append({
            "skill": skill,
            "allocated_level": alloc_level,
            "llm_judged_level": llm_level,
            "delta": delta,
            "flagged": delta >= 2,
        })

    return {
        "doc_id": document["doc_id"],
        "doc_type": document["type"],
        "comparisons": comparisons,
        "flagged_count": sum(1 for c in comparisons if c["flagged"]),
    }


async def check_allocation_rationality(
    persona: dict,
    doc_plans: list[dict],
    allocation: dict,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> dict:
    """LLM Check B: Assess if the evidence allocation makes sense.

    Returns dict with rationality score and concerns.
    """
    # Build documents list
    docs_lines = []
    for doc in doc_plans:
        title = doc.get("title", doc["doc_type"])
        topic = doc.get("topic_summary", "")
        docs_lines.append(f"- [{doc['doc_id']}] {doc['doc_type']}: \"{title}\" — {topic}")

    # Build skills list
    skills_lines = []
    for skill, level in sorted(persona["skills"].items()):
        skills_lines.append(f"- {skill}: Level {level}")

    prompt = VALIDATION_ALLOCATION_RATIONALITY.format(
        current_role=persona["current_role"],
        years_experience=persona["hyperparams"]["years_experience"],
        industry=persona["hyperparams"]["industry"],
        skills_list="\n".join(skills_lines),
        documents_list="\n".join(docs_lines),
        allocation_json=json.dumps(allocation, indent=2),
    )

    try:
        raw = await llm_call(session, semaphore, prompt)
        result = _extract_json(raw)
    except Exception as e:
        logger.warning(f"Allocation rationality check failed for {persona['persona_id']}: {e}")
        return {
            "persona_id": persona["persona_id"],
            "error": str(e),
            "score": 0,
            "concerns": [],
        }

    score = result.get("score", 0)
    concerns = result.get("concerns", [])

    return {
        "persona_id": persona["persona_id"],
        "score": score,
        "concerns": concerns,
        "flagged": score < 3,
    }


async def run_validation_checks(
    personas: list[dict],
    documents_db: dict,
    allocations_dir: Path,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
):
    """Run all LLM validation checks and write reports.

    Outputs:
        output_dir/skill_showcase_report.json
        output_dir/allocation_rationality_report.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Check A: Skill Showcase ---
    logger.info("Running LLM Check A: Per-document skill showcase...")
    showcase_tasks = []
    for doc_id, doc in documents_db.items():
        skills = list(doc.get("skill_evidence", {}).keys())
        # Only check skills that should appear (intensity > 1)
        skills_to_check = [
            s for s, v in doc.get("skill_evidence", {}).items() if v > 1
        ]
        if skills_to_check:
            showcase_tasks.append(
                check_skill_showcase(doc, skills_to_check, session, semaphore)
            )

    showcase_results = await asyncio.gather(*showcase_tasks)
    showcase_report = {
        "total_documents": len(showcase_results),
        "flagged_documents": sum(1 for r in showcase_results if r.get("flagged_count", 0) > 0),
        "results": showcase_results,
    }

    showcase_file = output_dir / "skill_showcase_report.json"
    with open(showcase_file, "w") as f:
        json.dump(showcase_report, f, indent=2)
    logger.info(f"Wrote skill showcase report: {showcase_file}")

    # --- Check B: Allocation Rationality ---
    logger.info("Running LLM Check B: Allocation rationality...")
    alloc_tasks = []
    for persona in personas:
        pid = persona["persona_id"]
        # Load doc plans and allocation from checkpoints
        docs_file = allocations_dir / f"{pid}_docs.json"
        alloc_file = allocations_dir / f"{pid}_alloc.json"

        if docs_file.exists() and alloc_file.exists():
            with open(docs_file) as f:
                doc_plans = json.load(f)
            with open(alloc_file) as f:
                allocation = json.load(f)
            alloc_tasks.append(
                check_allocation_rationality(
                    persona, doc_plans, allocation, session, semaphore
                )
            )

    alloc_results = await asyncio.gather(*alloc_tasks)
    alloc_report = {
        "total_personas": len(alloc_results),
        "flagged_personas": sum(1 for r in alloc_results if r.get("flagged", False)),
        "results": alloc_results,
    }

    alloc_file = output_dir / "allocation_rationality_report.json"
    with open(alloc_file, "w") as f:
        json.dump(alloc_report, f, indent=2)
    logger.info(f"Wrote allocation rationality report: {alloc_file}")

    return showcase_report, alloc_report

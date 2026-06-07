"""
Phase 3+4: Document planning and evidence allocation via LLM.
"""

import json
import asyncio
import aiohttp
import logging
from pathlib import Path

from generate.hyperparams import sample_doc_count, sample_document_hyperparams
from generate.prompts import DOCUMENT_PLANNING, EVIDENCE_ALLOCATION
from generate.personas import llm_call, _extract_json

logger = logging.getLogger(__name__)


def validate_allocation(allocation: dict, global_skills: dict) -> bool:
    """Validate evidence allocation against hard constraints.

    Rules:
    - For each skill, max(local_intensity) == global_level
    - No local intensity > global_level
    - Level 1 skills: all local intensities == 1
    """
    errors = []
    for skill, global_level in global_skills.items():
        if skill not in allocation:
            errors.append(f"{skill}: missing from allocation")
            continue

        local_values = list(allocation[skill].values())

        if not local_values:
            errors.append(f"{skill}: no document allocations")
            continue

        if global_level == 1:
            if not all(v == 1 for v in local_values):
                errors.append(f"{skill}: level 1 must have all 1s, got {local_values}")
        else:
            max_local = max(local_values)
            if max_local != global_level:
                errors.append(f"{skill}: max must equal {global_level}, got {max_local}")
            if any(v > global_level for v in local_values):
                errors.append(f"{skill}: values exceed global level {global_level}: {local_values}")
            if any(v < 1 for v in local_values):
                errors.append(f"{skill}: values below 1: {local_values}")

    if errors:
        for e in errors:
            logger.warning(f"Allocation error: {e}")
        return False
    return True


async def plan_documents(
    persona: dict,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    checkpoint_dir: Path,
) -> list[dict]:
    """Phase 3: Plan which documents to generate for a persona.

    Returns list of document plan dicts with type, topic, title, and hyperparams.
    """
    persona_id = persona["persona_id"]
    checkpoint_file = checkpoint_dir / f"{persona_id}_docs.json"

    if checkpoint_file.exists():
        logger.info(f"Skipping doc planning for {persona_id} — checkpoint exists")
        with open(checkpoint_file) as f:
            return json.load(f)

    doc_count = sample_doc_count(persona["hyperparams"]["years_experience"])

    prompt = DOCUMENT_PLANNING.format(
        name=persona["name"],
        current_role=persona["current_role"],
        company=persona["company"],
        years_experience=persona["hyperparams"]["years_experience"],
        industry=persona["hyperparams"]["industry"],
        archetype_label=persona["hyperparams"]["archetype"],
        doc_count=doc_count,
        remaining_count=doc_count - 1,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw = await llm_call(session, semaphore, prompt)
            doc_plans = _extract_json(raw)

            if not isinstance(doc_plans, list):
                raise ValueError("Expected a JSON list")

            # Validate: must have exactly 1 CV
            cv_count = sum(1 for d in doc_plans if d.get("doc_type") == "cv")
            if cv_count != 1:
                raise ValueError(f"Expected exactly 1 CV, got {cv_count}")

            if len(doc_plans) != doc_count:
                # Adjust if LLM gave different count — use what we got if reasonable
                if len(doc_plans) < 1 or len(doc_plans) > 7:
                    raise ValueError(f"Unreasonable doc count: {len(doc_plans)}")
                logger.warning(
                    f"{persona_id}: requested {doc_count} docs, got {len(doc_plans)}"
                )

            break
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"{persona_id} doc planning attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                # Fallback: create a minimal plan with just a CV
                doc_plans = [{"doc_type": "cv", "topic_summary": "Full CV", "title": "CV"}]

    # Assign IDs and sample document-level hyperparams
    for i, doc in enumerate(doc_plans):
        doc["doc_id"] = f"d_{persona_id}_{i:02d}"
        doc["persona_id"] = persona_id
        doc["hyperparams"] = sample_document_hyperparams(doc["doc_type"])

    # Checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(doc_plans, f, indent=2)

    logger.info(f"Planned {len(doc_plans)} documents for {persona_id}")
    return doc_plans


async def allocate_evidence(
    persona: dict,
    doc_plans: list[dict],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    checkpoint_dir: Path,
) -> dict:
    """Phase 4: Allocate per-document evidence intensities via LLM.

    Returns dict: {skill_name: {doc_id: intensity}}
    """
    persona_id = persona["persona_id"]
    checkpoint_file = checkpoint_dir / f"{persona_id}_alloc.json"

    if checkpoint_file.exists():
        logger.info(f"Skipping allocation for {persona_id} — checkpoint exists")
        with open(checkpoint_file) as f:
            return json.load(f)

    # Build documents list for prompt
    docs_list_lines = []
    for i, doc in enumerate(doc_plans, 1):
        title = doc.get("title", doc["doc_type"])
        topic = doc.get("topic_summary", "")
        docs_list_lines.append(
            f"{i}. [{doc['doc_id']}] {doc['doc_type']}: \"{title}\" — {topic}"
        )

    # Build skills list for prompt
    skills_lines = []
    for skill, level in sorted(persona["skills"].items()):
        skills_lines.append(f"- {skill}: Level {level}")

    prompt = EVIDENCE_ALLOCATION.format(
        name=persona["name"],
        current_role=persona["current_role"],
        years_experience=persona["hyperparams"]["years_experience"],
        industry=persona["hyperparams"]["industry"],
        documents_list="\n".join(docs_list_lines),
        skills_list="\n".join(skills_lines),
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw = await llm_call(session, semaphore, prompt)
            allocation = _extract_json(raw)

            if not isinstance(allocation, dict):
                raise ValueError("Expected a JSON dict")

            # Ensure all skills are present
            for skill in persona["skills"]:
                if skill not in allocation:
                    # Default: set all docs to 1 except one matching global level
                    allocation[skill] = {}
                    for doc in doc_plans:
                        allocation[skill][doc["doc_id"]] = 1
                    # Set first doc to global level (unless level 1)
                    if persona["skills"][skill] > 1:
                        allocation[skill][doc_plans[0]["doc_id"]] = persona["skills"][skill]

            # Ensure all doc IDs are present for each skill
            doc_ids = {doc["doc_id"] for doc in doc_plans}
            for skill in allocation:
                for doc_id in doc_ids:
                    if doc_id not in allocation[skill]:
                        allocation[skill][doc_id] = 1

            # Convert string values to int
            for skill in allocation:
                for doc_id in allocation[skill]:
                    allocation[skill][doc_id] = int(allocation[skill][doc_id])

            # Validate hard constraints
            if validate_allocation(allocation, persona["skills"]):
                break
            else:
                raise ValueError("Allocation failed hard constraints")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"{persona_id} allocation attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                # Fallback: deterministic allocation
                logger.error(f"{persona_id}: using fallback allocation")
                allocation = _fallback_allocation(persona, doc_plans)

    # Checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(allocation, f, indent=2)

    logger.info(f"Allocated evidence for {persona_id} ({len(persona['skills'])} skills × {len(doc_plans)} docs)")
    return allocation


def _fallback_allocation(persona: dict, doc_plans: list[dict]) -> dict:
    """Deterministic fallback allocation that always satisfies constraints."""
    allocation = {}
    doc_ids = [d["doc_id"] for d in doc_plans]

    for skill, global_level in persona["skills"].items():
        allocation[skill] = {}
        if global_level == 1:
            for doc_id in doc_ids:
                allocation[skill][doc_id] = 1
        else:
            # Put max evidence in the first doc (CV), 1 in others
            for i, doc_id in enumerate(doc_ids):
                if i == 0:
                    allocation[skill][doc_id] = global_level
                else:
                    allocation[skill][doc_id] = 1

    return allocation

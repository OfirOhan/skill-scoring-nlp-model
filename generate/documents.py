"""
Phase 5: Document text generation via LLM.
"""

import json
import asyncio
import aiohttp
import logging
from pathlib import Path

from spe.generate.prompts import (
    CV_GENERATION,
    README_GENERATION,
    RECOMMENDATION_GENERATION,
    LINKEDIN_GENERATION,
    BLOG_GENERATION,
    build_skill_evidence_instructions,
)
from spe.generate.personas import llm_call
from spe.generate.sanitizer import sanitize_document

logger = logging.getLogger(__name__)

# Map doc types to prompt templates
PROMPT_TEMPLATES = {
    "cv": CV_GENERATION,
    "project_readme": README_GENERATION,
    "recommendation": RECOMMENDATION_GENERATION,
    "linkedin": LINKEDIN_GENERATION,
    "blog": BLOG_GENERATION,
}


def _extract_text(raw: str) -> str:
    """Extract document text from LLM response, stripping thinking tags."""
    text = raw.strip()

    # Strip thinking tags if present
    if "<think>" in text:
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()

    # Strip markdown fences if the LLM wrapped the output
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    return text


async def generate_document(
    doc_plan: dict,
    persona: dict,
    allocation: dict,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    checkpoint_dir: Path,
) -> dict:
    """Generate a single document's text via LLM.

    Returns the complete document dict for documents_db.json.
    """
    doc_id = doc_plan["doc_id"]
    doc_type = doc_plan["doc_type"]
    checkpoint_file = checkpoint_dir / f"{doc_id}.json"

    if checkpoint_file.exists():
        logger.info(f"Skipping {doc_id} — checkpoint exists")
        with open(checkpoint_file) as f:
            return json.load(f)

    # Build skill evidence for this specific document
    doc_skill_evidence = {}
    for skill, doc_allocs in allocation.items():
        intensity = doc_allocs.get(doc_id, 1)
        doc_skill_evidence[skill] = intensity

    skill_evidence_instructions = build_skill_evidence_instructions(doc_skill_evidence)

    # Build the prompt based on document type
    hp = persona["hyperparams"]
    dp = doc_plan.get("hyperparams", {})

    prompt_kwargs = {
        "name": persona["name"],
        "current_role": persona["current_role"],
        "company": persona.get("company", ""),
        "years_experience": hp["years_experience"],
        "seniority": hp["seniority"],
        "industry": hp["industry"],
        "education": json.dumps(persona.get("education", {})),
        "career_trajectory": json.dumps(persona.get("career_trajectory", [])),
        "writing_style": hp["writing_style"],
        "language_fluency": hp["language_fluency"],
        "self_promotion_level": hp["self_promotion_level"],
        "quantification_tendency": hp["quantification_tendency"],
        "skill_evidence_instructions": skill_evidence_instructions,
    }

    # Add document-level params
    prompt_kwargs.update(dp)

    # Add doc-specific fields
    prompt_kwargs["title"] = doc_plan.get("title", "")
    prompt_kwargs["topic_summary"] = doc_plan.get("topic_summary", "")

    # Select template and fill
    template = PROMPT_TEMPLATES.get(doc_type)
    if not template:
        raise ValueError(f"Unknown doc type: {doc_type}")

    # Only include kwargs that are in the template
    prompt = template
    for key, val in prompt_kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, str(val))

    # Generate with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw = await llm_call(session, semaphore, prompt)
            text = _extract_text(raw)

            if len(text) < 50:
                raise ValueError(f"Generated text too short: {len(text)} chars")

            # Sanitize
            issues = sanitize_document(text, doc_skill_evidence)
            if issues:
                logger.warning(f"{doc_id}: sanitization issues: {issues}")
                if attempt < max_retries - 1:
                    continue  # Retry

            break
        except Exception as e:
            logger.warning(f"{doc_id} generation attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                text = f"[GENERATION FAILED: {e}]"

    # Assemble document record
    document = {
        "doc_id": doc_id,
        "type": doc_type,
        "persona_id": persona["persona_id"],
        "hyperparams": dp,
        "skill_evidence": doc_skill_evidence,
        "text": text,
    }

    # Checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated {doc_id} ({doc_type}, {len(text)} chars)")
    return document

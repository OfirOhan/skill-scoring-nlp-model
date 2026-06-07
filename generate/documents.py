"""
Phase 5: Document text generation via LLM.

Integrates:
- Adaptive banned phrase tracking (TF-IDF-based)
- Document structure seeds
- Per-document temperature sampling
"""

import json
import asyncio
import aiohttp
import logging
import random
from pathlib import Path

from generate.prompts import (
    CV_GENERATION,
    README_GENERATION,
    RECOMMENDATION_GENERATION,
    LINKEDIN_GENERATION,
    BLOG_GENERATION,
    build_skill_evidence_instructions,
)
from generate.personas import llm_call
from generate.sanitizer import sanitize_document
from generate.phrase_tracker import PhraseTracker
from generate.seed_generator import pick_seed, format_seed_for_prompt

logger = logging.getLogger(__name__)

# Map doc types to prompt templates
PROMPT_TEMPLATES = {
    "cv": CV_GENERATION,
    "project_readme": README_GENERATION,
    "recommendation": RECOMMENDATION_GENERATION,
    "linkedin": LINKEDIN_GENERATION,
    "blog": BLOG_GENERATION,
}

# Temperature ranges per document type.
# Text-heavy documents benefit from higher temperature for diversity.
TEMPERATURE_RANGES = {
    "cv": (0.75, 1.0),
    "project_readme": (0.7, 0.95),
    "recommendation": (0.85, 1.1),
    "linkedin": (0.8, 1.05),
    "blog": (0.8, 1.05),
}

# Module-level phrase tracker — shared across all document generation calls.
# Safe for asyncio (single-threaded event loop).
_phrase_tracker = PhraseTracker()


def get_phrase_tracker() -> PhraseTracker:
    """Get the module-level phrase tracker (for testing/inspection)."""
    return _phrase_tracker


def _sample_temperature(doc_type: str) -> float:
    """Sample a temperature for this document type from its configured range."""
    lo, hi = TEMPERATURE_RANGES.get(doc_type, (0.8, 1.0))
    return round(random.uniform(lo, hi), 2)


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
    structure_seeds: dict | None = None,
) -> dict:
    """Generate a single document's text via LLM.

    Args:
        structure_seeds: Pre-generated structure seeds dict (doc_type -> list of seeds).
                         If None, no structural seed is injected.

    Returns the complete document dict for documents_db.json.
    """
    doc_id = doc_plan["doc_id"]
    doc_type = doc_plan["doc_type"]
    checkpoint_file = checkpoint_dir / f"{doc_id}.json"

    if checkpoint_file.exists():
        logger.info(f"Skipping {doc_id} — checkpoint exists")
        with open(checkpoint_file) as f:
            doc = json.load(f)
        # Still ingest into phrase tracker for adaptive banning
        _phrase_tracker.ingest(doc.get("text", ""))
        return doc

    # Build skill evidence for this specific document
    doc_skill_evidence = {}
    for skill, doc_allocs in allocation.items():
        intensity = doc_allocs.get(doc_id, 1)
        doc_skill_evidence[skill] = intensity

    skill_evidence_instructions = build_skill_evidence_instructions(doc_skill_evidence)

    # Build the prompt based on document type
    hp = persona["hyperparams"]
    dp = doc_plan.get("hyperparams", {})

    # --- Structure seed ---
    seed_text = ""
    if structure_seeds:
        seniority = hp.get("seniority", "all")
        seed = pick_seed(structure_seeds, doc_type, seniority)
        seed_text = format_seed_for_prompt(seed)
    if not seed_text:
        seed_text = "Use a natural structure appropriate for this document type."

    # --- Adaptive banned phrases ---
    banned_phrases_text = _phrase_tracker.format_for_prompt(top_k=10)

    # --- Temperature ---
    temperature = _sample_temperature(doc_type)

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
        "structure_seed": seed_text,
        "banned_phrases": banned_phrases_text,
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
            raw = await llm_call(session, semaphore, prompt, temperature=temperature)
            text = _extract_text(raw)

            if len(text) < 50:
                raise ValueError(f"Generated text too short: {len(text)} chars")

            # Sanitize
            issues = sanitize_document(text, doc_skill_evidence)
            if issues:
                logger.warning(f"{doc_id}: sanitization issues: {issues}")
                if attempt < max_retries - 1:
                    # Bump temperature slightly on retry for more diversity
                    temperature = min(temperature + 0.1, 1.3)
                    continue  # Retry

            break
        except Exception as e:
            logger.warning(f"{doc_id} generation attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                text = f"[GENERATION FAILED: {e}]"

    # Ingest into phrase tracker for future documents
    _phrase_tracker.ingest(text)

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

    logger.info(
        f"Generated {doc_id} ({doc_type}, {len(text)} chars, "
        f"temp={temperature}, banned={len(_phrase_tracker.get_banned_phrases())})"
    )
    return document

"""
Phase 2: Persona generation via LLM.
"""

import json
import asyncio
import aiohttp
import logging
from pathlib import Path

from spe.generate.hyperparams import (
    sample_archetype,
    sample_persona_hyperparams,
    seniority_constraints,
    get_archetypes,
    get_skills_by_category,
)
from spe.generate.prompts import PERSONA_GENERATION

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3"


async def llm_call(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    prompt: str,
) -> str:
    """Make an async LLM call to ollama with concurrency control."""
    async with semaphore:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 4096},
        }
        async with session.post(OLLAMA_URL, json=payload) as resp:
            result = await resp.json()
            return result["message"]["content"].strip()


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and thinking tags."""
    # Strip thinking tags if present
    if "<think>" in text:
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()

    # Strip markdown fences
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    return json.loads(text)


async def generate_persona(
    persona_id: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    checkpoint_dir: Path,
) -> dict:
    """Generate a single persona with skills via LLM.

    Returns the persona dict and saves a checkpoint.
    """
    # Check for existing checkpoint
    checkpoint_file = checkpoint_dir / f"{persona_id}.json"
    if checkpoint_file.exists():
        logger.info(f"Skipping {persona_id} — checkpoint exists")
        with open(checkpoint_file) as f:
            return json.load(f)

    # Sample hyperparameters
    archetype_key = sample_archetype()
    hyperparams = sample_persona_hyperparams(archetype_key)
    archetypes = get_archetypes()
    archetype = archetypes[archetype_key]

    # Build skill pool for the prompt
    primary_skills = []
    for cat in archetype["primary_categories"]:
        primary_skills.extend(get_skills_by_category(cat))

    secondary_skills = []
    for cat in archetype["secondary_categories"]:
        secondary_skills.extend(get_skills_by_category(cat))

    # Determine skill counts
    import random
    total = random.randint(*archetype["skill_count_range"])
    primary_count = max(4, int(total * 0.6))
    secondary_count = total - primary_count

    # Build the prompt
    prompt = PERSONA_GENERATION.format(
        archetype_label=archetype["label"],
        primary_categories=", ".join(archetype["primary_categories"]),
        years_experience=hyperparams["years_experience"],
        seniority=hyperparams["seniority"],
        industry=hyperparams["industry"],
        primary_skills="\n".join(f"  - {s}" for s in primary_skills),
        secondary_skills="\n".join(f"  - {s}" for s in secondary_skills),
        primary_count=primary_count,
        secondary_count=secondary_count,
        total_skill_count=total,
        seniority_constraints=seniority_constraints(hyperparams["seniority"]),
        writing_style=hyperparams["writing_style"],
        language_fluency=hyperparams["language_fluency"],
        self_promotion_level=hyperparams["self_promotion_level"],
        quantification_tendency=hyperparams["quantification_tendency"],
    )

    # Call LLM with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw = await llm_call(session, semaphore, prompt)
            persona_data = _extract_json(raw)

            # Validate required fields
            required = ["name", "current_role", "company", "education", "skills"]
            for field in required:
                if field not in persona_data:
                    raise ValueError(f"Missing field: {field}")

            # Validate skill levels
            for skill, level in persona_data["skills"].items():
                if not isinstance(level, int) or level < 1 or level > 5:
                    raise ValueError(f"Invalid level for {skill}: {level}")

            break
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"{persona_id} attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate {persona_id} after {max_retries} attempts") from e

    # Assemble final persona
    persona = {
        "persona_id": persona_id,
        "hyperparams": hyperparams,
        "name": persona_data["name"],
        "current_role": persona_data["current_role"],
        "company": persona_data["company"],
        "education": persona_data.get("education", {}),
        "career_trajectory": persona_data.get("career_trajectory", []),
        "skills": persona_data["skills"],
        "document_ids": [],  # filled in Phase 3
    }

    # Save checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(persona, f, indent=2)

    logger.info(f"Generated {persona_id}: {persona['name']} ({archetype['label']}, {hyperparams['years_experience']}yr)")
    return persona

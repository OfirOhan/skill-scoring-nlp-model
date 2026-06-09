"""
Phase 2: Persona generation via LLM.
"""

import json
import asyncio
import aiohttp
import logging
import random as _random
from pathlib import Path

from generate.hyperparams import (
    sample_archetype,
    sample_persona_hyperparams,
    seniority_band_bias,
    low_level_quota,
    get_archetypes,
    get_skills_by_category,
)
from generate.prompts import PERSONA_GENERATION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name pool — ensures diverse, non-repeating names across personas
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_name_pool: list[str] | None = None
_used_names: set[str] = set()


def _load_name_pool() -> list[str]:
    """Load and shuffle the curated name pool."""
    global _name_pool
    if _name_pool is None:
        pool_path = DATA_DIR / "name_pool.json"
        with open(pool_path) as f:
            _name_pool = json.load(f)
        _random.shuffle(_name_pool)
    return _name_pool


def pick_unique_name() -> str:
    """Pick a name from the pool that hasn't been used yet.

    Falls back to appending a numeric suffix if the pool is exhausted.
    """
    pool = _load_name_pool()
    for name in pool:
        if name not in _used_names:
            _used_names.add(name)
            return name
    # Pool exhausted — generate a suffixed name
    base = _random.choice(pool)
    suffix = len(_used_names) - len(pool) + 1
    unique = f"{base} {suffix}"
    _used_names.add(unique)
    return unique

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3"


async def llm_call(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    prompt: str,
    temperature: float = 0.8,
) -> str:
    """Make an async LLM call to ollama with concurrency control.

    Args:
        temperature: Sampling temperature. Higher values (0.9-1.1) produce
                     more diverse text, lower values (0.6-0.8) are more focused.
    """
    async with semaphore:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 4096},
        }
        async with session.post(OLLAMA_URL, json=payload) as resp:
            result = await resp.json()
            return result["message"]["content"].strip()


def enforce_skill_levels(
    skills: dict,
    primary_pool: set[str],
    secondary_pool: set[str],
    primary_band: tuple[int, int],
    secondary_band: tuple[int, int],
    seniority: str,
) -> tuple[dict, dict]:
    """Deterministic guard: force the LLM's skill levels onto the tier scheme.

    Two guarantees, applied after generation so the label distribution can never
    silently collapse again:

      1. **Band compliance** — primary skills are clamped into `primary_band`,
         secondary skills into `secondary_band`. A primary skill rated 1 becomes
         the band floor; a secondary rated 5 becomes the band ceiling.
      2. **Low-level floor** — at least `low_level_quota()` secondary skills must
         end up at level <= 2 (split across 1 and 2). If the LLM rated every
         breadth skill at 3, the highest secondaries are demoted to fill the
         quota. Only the *secondary* (peripheral) tier is ever demoted, so the
         profile stays rational.

    Returns (fixed_skills, tier_of_skill).
    """
    p_lo, p_hi = primary_band
    s_lo, s_hi = secondary_band

    fixed: dict = {}
    tier: dict = {}
    for sk, lv in skills.items():
        lv = int(lv)
        if sk in primary_pool:
            tier[sk] = "primary"
            fixed[sk] = max(p_lo, min(p_hi, lv))
        else:
            # secondary pool, soft-skills, or anything off-pool -> treat as breadth
            tier[sk] = "secondary"
            fixed[sk] = max(s_lo, min(s_hi, lv))

    secondary = [sk for sk in fixed if tier[sk] == "secondary"]
    if secondary:
        target_low = low_level_quota(len(secondary), seniority)
        target_one = max(1, target_low // 3)  # a few genuine level-1s
        have_low = sum(1 for sk in secondary if fixed[sk] <= 2)

        # Demote the highest-rated secondary skills first (least disruptive).
        for sk in sorted(secondary, key=lambda s: fixed[s], reverse=True):
            if have_low >= target_low:
                break
            if fixed[sk] > 2:
                fixed[sk] = 2
                have_low += 1
        # Ensure some land at exactly 1 (Awareness), not just 2.
        ones = sum(1 for sk in secondary if fixed[sk] == 1)
        for sk in sorted(secondary, key=lambda s: fixed[s]):
            if ones >= target_one:
                break
            if fixed[sk] == 2:
                fixed[sk] = 1
                ones += 1

    return fixed, tier


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

    # Pick a unique name from the diversity pool
    pre_selected_name = pick_unique_name()

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

    # Per-tier level bands (these archetype fields were previously unused — they
    # are the rational level scheme that keeps low levels on breadth skills).
    primary_band = tuple(archetype.get("primary_skill_levels", [3, 5]))
    secondary_band = tuple(archetype.get("secondary_skill_levels", [1, 3]))
    primary_pool = set(primary_skills)
    secondary_pool = set(secondary_skills)

    # Determine skill counts
    import random
    total = random.randint(*archetype["skill_count_range"])
    primary_count = max(4, int(total * 0.6))
    secondary_count = total - primary_count

    # Build the prompt
    prompt = PERSONA_GENERATION.format(
        pre_selected_name=pre_selected_name,
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
        primary_band=f"{primary_band[0]}-{primary_band[1]}",
        secondary_band=f"{secondary_band[0]}-{secondary_band[1]}",
        seniority_bias=seniority_band_bias(hyperparams["seniority"]),
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

            # Deterministic guard: clamp to tier bands + enforce the low-level
            # floor so the global label distribution cannot collapse.
            original = dict(persona_data["skills"])
            persona_data["skills"], _ = enforce_skill_levels(
                persona_data["skills"], primary_pool, secondary_pool,
                primary_band, secondary_band, hyperparams["seniority"],
            )
            n_adjusted = sum(1 for k in original
                             if original[k] != persona_data["skills"].get(k))
            if n_adjusted:
                logger.info(f"{persona_id}: enforced tier bands on {n_adjusted}/"
                            f"{len(original)} skill levels")

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

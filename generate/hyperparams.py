"""
Hyperparameter sampling logic for persona and document generation.
"""

import random
import math
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Load static data
# ---------------------------------------------------------------------------

def load_skills_taxonomy() -> list[dict]:
    with open(DATA_DIR / "skills_taxonomy.json") as f:
        return json.load(f)


def load_archetypes() -> dict:
    with open(DATA_DIR / "archetypes.json") as f:
        return json.load(f)


def load_evidence_profiles() -> dict:
    with open(DATA_DIR / "evidence_profiles.json") as f:
        return json.load(f)


# Cache on first load
_taxonomy = None
_archetypes = None
_evidence_profiles = None


def get_taxonomy() -> list[dict]:
    global _taxonomy
    if _taxonomy is None:
        _taxonomy = load_skills_taxonomy()
    return _taxonomy


def get_archetypes() -> dict:
    global _archetypes
    if _archetypes is None:
        _archetypes = load_archetypes()
    return _archetypes


def get_evidence_profiles() -> dict:
    global _evidence_profiles
    if _evidence_profiles is None:
        _evidence_profiles = load_evidence_profiles()
    return _evidence_profiles


# ---------------------------------------------------------------------------
# Experience distribution (matches real-life hi-tech workforce)
# ---------------------------------------------------------------------------

EXPERIENCE_DISTRIBUTION = [
    # (min_years, max_years, weight)
    (0, 1, 15),   # Entry
    (2, 3, 20),   # Junior
    (4, 6, 25),   # Mid
    (7, 9, 20),   # Senior
    (10, 12, 12), # Staff
    (13, 15, 8),  # Principal
]


def sample_years_experience() -> int:
    """Sample years of experience from the weighted distribution."""
    brackets = []
    weights = []
    for min_y, max_y, w in EXPERIENCE_DISTRIBUTION:
        brackets.append((min_y, max_y))
        weights.append(w)
    bracket = random.choices(brackets, weights=weights, k=1)[0]
    return random.randint(bracket[0], bracket[1])


def years_to_seniority(years: int) -> str:
    if years <= 2:
        return "junior"
    elif years <= 5:
        return "mid"
    elif years <= 9:
        return "senior"
    else:
        return "staff"


def seniority_constraints(seniority: str) -> str:
    """DEPRECATED — kept for backward compatibility.

    These absolute floors collapsed the global skill-level distribution (no
    level-1 skills, everything 3-5) because they applied one band to ALL skills
    regardless of tier and never *required* low levels. Persona generation now
    uses per-tier bands (archetype.primary_skill_levels / secondary_skill_levels)
    plus seniority_band_bias() to modulate WITHIN those bands. Do not reintroduce.
    """
    constraints = {
        "junior": "Mostly levels 1-3 (max 1-2 skills at level 4, no level 5)",
        "mid": "Mostly levels 2-4 (max 1 skill at level 5, some level 1s allowed)",
        "senior": "Levels 2-5 (multiple level 4-5 skills expected, few level 1s)",
        "staff": "Levels 3-5 (multiple level 5 skills expected, few below level 3)",
    }
    return constraints.get(seniority, constraints["mid"])


def seniority_band_bias(seniority: str) -> str:
    """How seniority shifts skill levels *within* each tier's band.

    Crucially this modulates inside the per-tier bands rather than overriding
    them — so secondary (breadth) skills keep producing genuine level 1-2s for
    every seniority, while seniority still moves the centre of mass.
    """
    biases = {
        "junior": "Lean to the LOW end of each band: primary skills mostly at the "
                  "band minimum, and most secondary skills at 1-2.",
        "mid":    "Sit mid-band: primary skills around the middle of their band, "
                  "secondary skills spread across their full band.",
        "senior": "Lean to the HIGH end for primary skills (several at the band max); "
                  "secondary skills still include several genuine 1-2s.",
        "staff":  "Primary skills at the high end (multiple at the maximum); secondary "
                  "skills are breadth areas, so keep several at 1-2.",
    }
    return biases.get(seniority, biases["mid"])


def low_level_quota(n_secondary: int, seniority: str) -> int:
    """Minimum number of secondary skills that must end up at level <= 2.

    A deterministic floor enforced after generation (see personas.enforce_skill_levels)
    so the low end of the label distribution can never collapse again, even if the
    LLM rates every breadth skill at 3. Scaled by seniority — juniors have touched
    more things shallowly than staff.
    """
    frac = {"junior": 0.6, "mid": 0.5, "senior": 0.4, "staff": 0.35}.get(seniority, 0.5)
    return math.ceil(frac * n_secondary) if n_secondary else 0


# ---------------------------------------------------------------------------
# Persona-level hyperparameters
# ---------------------------------------------------------------------------

INDUSTRIES = [
    "fintech", "healthtech", "e-commerce", "adtech", "cybersecurity",
    "gaming", "SaaS", "enterprise", "startup", "consulting",
]

WRITING_STYLES = ["modest", "neutral", "confident", "verbose", "concise"]
LANGUAGE_FLUENCIES = ["native", "near-native", "ESL"]
SELF_PROMOTION_LEVELS = ["humble", "balanced", "boastful"]
QUANTIFICATION_TENDENCIES = ["metrics-heavy", "qualitative", "mixed"]


def sample_persona_hyperparams(archetype_key: str) -> dict:
    """Sample all persona-level hyperparameters."""
    years = sample_years_experience()
    seniority = years_to_seniority(years)

    return {
        "archetype": archetype_key,
        "years_experience": years,
        "seniority": seniority,
        "industry": random.choice(INDUSTRIES),
        "writing_style": random.choice(WRITING_STYLES),
        "language_fluency": random.choices(
            LANGUAGE_FLUENCIES, weights=[50, 30, 20], k=1
        )[0],
        "self_promotion_level": random.choice(SELF_PROMOTION_LEVELS),
        "quantification_tendency": random.choice(QUANTIFICATION_TENDENCIES),
    }


# ---------------------------------------------------------------------------
# Document-level hyperparameters
# ---------------------------------------------------------------------------

CV_FORMATS = ["bullet-heavy", "paragraph", "hybrid", "minimal"]
GEOGRAPHIC_CONVENTIONS = ["US", "Europe", "Israel", "India"]
COMPANY_TYPES = ["FAANG", "startup", "enterprise", "agency", "academic"]
COMPANY_SIZES = ["small (<50)", "mid (50-500)", "large (500+)"]
PROJECT_SCALES = ["hobby", "academic", "team-internal", "production"]
PROJECT_TEAM_SIZES = ["solo", "small (2-4)", "medium (5-15)", "large (15+)"]
JARGON_DENSITIES = ["heavy", "moderate", "plain"]
RECOMMENDER_ROLES = ["direct-manager", "skip-level", "peer", "professor"]
RECOMMENDER_CLOSENESS = ["worked-daily", "occasional", "brief"]
BLOG_DEPTHS = ["tutorial", "deep-dive", "opinion-piece"]
LINKEDIN_COMPLETENESS = ["minimal", "standard", "detailed"]


def sample_document_hyperparams(doc_type: str) -> dict:
    """Sample document-level hyperparameters based on document type."""
    params = {}

    if doc_type == "cv":
        params["cv_format"] = random.choice(CV_FORMATS)
        params["geographic_convention"] = random.choice(GEOGRAPHIC_CONVENTIONS)
        params["company_type"] = random.choice(COMPANY_TYPES)
        params["company_size"] = random.choice(COMPANY_SIZES)

    elif doc_type == "project_readme":
        params["project_scale"] = random.choice(PROJECT_SCALES)
        params["project_team_size"] = random.choice(PROJECT_TEAM_SIZES)
        params["jargon_density"] = random.choice(JARGON_DENSITIES)

    elif doc_type == "recommendation":
        params["recommender_role"] = random.choice(RECOMMENDER_ROLES)
        params["recommender_closeness"] = random.choice(RECOMMENDER_CLOSENESS)
        params["company_type"] = random.choice(COMPANY_TYPES)
        params["company_size"] = random.choice(COMPANY_SIZES)

    elif doc_type == "linkedin":
        params["linkedin_completeness"] = random.choice(LINKEDIN_COMPLETENESS)

    elif doc_type == "blog":
        params["blog_depth"] = random.choice(BLOG_DEPTHS)
        params["jargon_density"] = random.choice(JARGON_DENSITIES)

    return params


# ---------------------------------------------------------------------------
# Document count distribution
# ---------------------------------------------------------------------------

def sample_doc_count(years_experience: int) -> int:
    """Sample document count based on experience level.

    Low/mid (0-6yr): Normal μ=3, σ=1, clipped [1, 5]
    High (7+yr): Normal μ=5, σ=1, clipped [3, 7]
    """
    if years_experience <= 6:
        count = round(random.gauss(3, 1))
        return max(1, min(5, count))
    else:
        count = round(random.gauss(5, 1))
        return max(3, min(7, count))


# ---------------------------------------------------------------------------
# Archetype-weighted sampling
# ---------------------------------------------------------------------------

def sample_archetype() -> str:
    """Sample an archetype key weighted by the target distribution."""
    archetypes = get_archetypes()
    keys = list(archetypes.keys())
    weights = [archetypes[k]["percentage"] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def get_skills_by_category(category: str) -> list[str]:
    """Return all skill names belonging to a category."""
    return [s["skill"] for s in get_taxonomy() if s["category"] == category]

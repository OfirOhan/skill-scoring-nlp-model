"""
Pre-generation step: Generate document structure seeds.

Creates diverse section orderings (seeds) for each document type using
the local LLM. These seeds are then assigned to individual documents
during generation, ensuring structural variance.

Seeds are cached to disk after generation so this step only runs once.
"""

import json
import asyncio
import aiohttp
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3"

SEED_GENERATION_PROMPT = """\
You are designing diverse document structures for synthetic professional documents.

== TASK ==
Generate {num_seeds} DIFFERENT section orderings (seeds) for a {doc_type} document.
Each seed is an ordered list of section headings that the document MUST follow.

== DOCUMENT TYPE: {doc_type_label} ==
{doc_type_description}

== SENIORITY CONSTRAINTS ==
{seniority_constraints}

== RULES ==
1. Each seed MUST be a realistic ordering of sections for this document type
2. Seeds MUST differ meaningfully from each other — not just minor reorderings
3. Some sections are near-universal (marked with *), others are optional
4. Vary which optional sections are included/excluded across seeds
5. The section names should be the EXACT headings to use in the document
6. Each seed should have {min_sections}-{max_sections} sections
7. Be creative with formatting: some seeds may use sub-sections, some flat

Output a JSON object with this structure:
{{
  "seeds": [
    {{
      "id": "seed_1",
      "sections": ["Section Name 1", "Section Name 2", ...],
      "style_note": "Brief note on the overall feel of this layout"
    }},
    ...
  ]
}}

Output ONLY valid JSON, no markdown fences, no explanation.\
"""

DOC_TYPE_CONFIGS = {
    "cv": {
        "doc_type_label": "CV / Resume",
        "doc_type_description": """A professional CV or resume for a hi-tech worker.
Near-universal sections (*):
  * Name & Contact Info (always first)
  * Professional Experience / Work History
  * Education
Common optional sections:
  - Professional Summary / Profile
  - Technical Skills / Technologies
  - Projects / Side Projects
  - Certifications
  - Languages
  - Publications
  - Awards & Achievements
  - Volunteer / Open Source
  - Strengths / Core Competencies

IMPORTANT: The ordering and presence of sections varies significantly.
Some CVs lead with a summary, others jump straight to experience.
Some put skills at the top, others embed them in experience.
Some separate projects from work, others don't.""",
        "min_sections": 3,
        "max_sections": 8,
        "num_seeds": 10,
        "seniority_variants": {
            "junior": "Junior (0-2 years): May include coursework, internships, or academic projects. Education is often prominent. May lack a summary. Skills section common.",
            "mid": "Mid-level (3-5 years): Has 2-3 real work experiences. Summary optional. Skills can be inline or separate. Projects section possible.",
            "senior": "Senior (6-9 years): Strong experience section with 3+ roles. Summary likely. May include mentoring, architecture, or leadership sections.",
            "staff": "Staff+ (10+ years): Extensive experience. Likely has summary. May include publications, talks, patents, or advisory roles. Education less prominent.",
        },
    },
    "project_readme": {
        "doc_type_label": "GitHub Project README",
        "doc_type_description": """A GitHub-style project README in markdown.
Near-universal sections (*):
  * Project Title + Description
Common optional sections:
  - Badges (CI, coverage, license)
  - Tech Stack / Built With
  - Architecture / Design
  - Getting Started / Installation
  - Usage / Examples
  - API Reference
  - Configuration
  - Deployment
  - Contributing
  - Testing
  - Roadmap / Future Work
  - License
  - Acknowledgments

READMEs vary enormously: some are minimal (title + description + install),
others are comprehensive multi-section documents.""",
        "min_sections": 3,
        "max_sections": 9,
        "num_seeds": 15,
        "seniority_variants": {
            "all": "Any seniority level. Vary the professionalism and completeness.",
        },
    },
    "recommendation": {
        "doc_type_label": "Professional Recommendation Letter",
        "doc_type_description": """A professional recommendation or reference letter.
Recommendation letters are typically flowing prose, NOT sectioned documents.
However, they follow implicit structural patterns:

Possible structural patterns:
  - Formal letter: Salutation → Relationship context → Key strengths → Specific anecdotes → Summary → Closing
  - Anecdote-first: Specific story → Who they are → Broader strengths → Endorsement
  - Skill-focused: Introduction → Technical abilities → Soft skills → Team impact → Recommendation
  - Brief endorsement: Quick intro → 2-3 key points → Strong closing
  - Professor-style: Academic context → Research ability → Character → Potential
  - Peer review: Working relationship → Day-to-day observations → Standout moments → Recommendation

Each seed should describe the PARAGRAPH FLOW, not section headers.""",
        "min_sections": 3,
        "max_sections": 6,
        "num_seeds": 8,
        "seniority_variants": {
            "all": "Any seniority level. Vary the recommender perspective.",
        },
    },
    "linkedin": {
        "doc_type_label": "LinkedIn About/Summary",
        "doc_type_description": """A LinkedIn About/Summary section.
These are typically 1-4 paragraphs of flowing text.

Possible structural patterns:
  - Mission-driven: Opens with passion/mission → Current work → Key achievements → What I'm looking for
  - Career narrative: Journey from X to Y → Current focus → Skills highlight → Call to action
  - Achievement-led: Headline number/achievement → Context → Broader expertise → Invitation to connect
  - Third-person bio: Written about the person, not by them. Formal. Factual.
  - Minimalist: 2-3 punchy sentences. No fluff.
  - Hashtag-style: Ends with hashtags/keywords. More casual.

Each seed should describe the FLOW, not section headers.""",
        "min_sections": 2,
        "max_sections": 5,
        "num_seeds": 6,
        "seniority_variants": {
            "all": "Any seniority level.",
        },
    },
    "blog": {
        "doc_type_label": "Technical Blog Post",
        "doc_type_description": """A technical blog post or article excerpt.
Common structural patterns:
  - Tutorial: Problem → Setup → Step-by-step → Results → Conclusion
  - Deep-dive: Hook → Background → Core analysis → Benchmarks → Takeaways
  - Opinion piece: Thesis → Supporting arguments → Counter-arguments → Conclusion
  - Case study: Context → Challenge → Solution → Results → Lessons learned
  - Listicle: Introduction → Numbered items → Summary
  - Problem-solution: Problem statement → Failed approaches → Working solution → Why it works

Each seed describes the article flow.""",
        "min_sections": 3,
        "max_sections": 7,
        "num_seeds": 6,
        "seniority_variants": {
            "all": "Any seniority level.",
        },
    },
}


async def _llm_call(session: aiohttp.ClientSession, prompt: str) -> str:
    """Make an LLM call for seed generation."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 1.0, "num_predict": 4096},
    }
    async with session.post(OLLAMA_URL, json=payload) as resp:
        result = await resp.json()
        return result["message"]["content"].strip()


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    if "<think>" in text:
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
    return json.loads(text)


async def generate_seeds_for_type(
    doc_type: str,
    session: aiohttp.ClientSession,
) -> list[dict]:
    """Generate structure seeds for a single document type."""
    config = DOC_TYPE_CONFIGS[doc_type]
    all_seeds = []

    for seniority_key, seniority_desc in config["seniority_variants"].items():
        prompt = SEED_GENERATION_PROMPT.format(
            num_seeds=config["num_seeds"],
            doc_type=doc_type,
            doc_type_label=config["doc_type_label"],
            doc_type_description=config["doc_type_description"],
            seniority_constraints=seniority_desc,
            min_sections=config["min_sections"],
            max_sections=config["max_sections"],
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = await _llm_call(session, prompt)
                result = _extract_json(raw)
                seeds = result.get("seeds", [])

                if not seeds:
                    raise ValueError("No seeds in response")

                # Tag each seed with its seniority and doc type
                for seed in seeds:
                    seed["doc_type"] = doc_type
                    seed["seniority"] = seniority_key

                all_seeds.extend(seeds)
                logger.info(
                    f"Generated {len(seeds)} seeds for {doc_type}/{seniority_key}"
                )
                break
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    f"Seed generation {doc_type}/{seniority_key} "
                    f"attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to generate seeds for {doc_type}/{seniority_key}"
                    )

    return all_seeds


async def generate_all_seeds(
    cache_path: Path,
    force: bool = False,
) -> dict[str, list[dict]]:
    """Generate structure seeds for all document types.

    Seeds are cached to disk. Pass force=True to regenerate.

    Returns:
        Dict mapping doc_type -> list of seed dicts.
    """
    if cache_path.exists() and not force:
        logger.info(f"Loading cached seeds from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    logger.info("Generating document structure seeds...")

    all_seeds = {}
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        for doc_type in DOC_TYPE_CONFIGS:
            seeds = await generate_seeds_for_type(doc_type, session)
            all_seeds[doc_type] = seeds
            logger.info(f"Total seeds for {doc_type}: {len(seeds)}")

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_seeds, f, indent=2, ensure_ascii=False)

    logger.info(f"All seeds cached to {cache_path}")
    return all_seeds


def pick_seed(
    seeds: dict[str, list[dict]],
    doc_type: str,
    seniority: str = "all",
) -> dict:
    """Pick a random seed for a given doc type and seniority.

    Falls back to 'all' seniority if no seniority-specific seeds exist.
    """
    type_seeds = seeds.get(doc_type, [])

    if not type_seeds:
        # Fallback: return a generic seed
        return {
            "id": "fallback",
            "sections": [],
            "style_note": "No seed available — use default structure",
            "doc_type": doc_type,
            "seniority": seniority,
        }

    # Filter by seniority if applicable
    seniority_seeds = [s for s in type_seeds if s.get("seniority") == seniority]
    if not seniority_seeds:
        seniority_seeds = [s for s in type_seeds if s.get("seniority") == "all"]
    if not seniority_seeds:
        seniority_seeds = type_seeds

    return random.choice(seniority_seeds)


def format_seed_for_prompt(seed: dict) -> str:
    """Format a seed as a prompt-ready instruction block.

    Returns a string to inject into the generation prompt.
    """
    sections = seed.get("sections", [])
    style_note = seed.get("style_note", "")

    if not sections:
        return ""

    lines = []
    lines.append(f"Document layout style: {style_note}")
    lines.append("You MUST use these sections IN THIS EXACT ORDER:")
    for i, section in enumerate(sections, 1):
        lines.append(f"  {i}. {section}")
    lines.append("Do NOT add extra sections. Do NOT reorder.")

    return "\n".join(lines)

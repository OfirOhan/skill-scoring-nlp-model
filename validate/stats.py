"""
Validation: Distribution and correlation statistics.
All checks are read-only — output reports only.
"""

import json
import logging
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)


def check_proficiency_distribution(personas: list[dict], documents_db: dict) -> dict:
    """Check that all 5 proficiency levels have >= 12% representation."""
    level_counts = Counter()

    for persona in personas:
        for doc_id in persona.get("document_ids", []):
            doc = documents_db.get(doc_id)
            if not doc:
                continue
            for skill, intensity in doc.get("skill_evidence", {}).items():
                level_counts[intensity] += 1

    total = sum(level_counts.values())
    distribution = {}
    for level in range(1, 6):
        count = level_counts.get(level, 0)
        pct = (count / total * 100) if total > 0 else 0
        distribution[level] = {
            "count": count,
            "percentage": round(pct, 1),
            "passed": pct >= 12,
        }

    return {
        "total_examples": total,
        "distribution": distribution,
        "all_passed": all(d["passed"] for d in distribution.values()),
    }


def check_global_level_distribution(personas: list[dict]) -> dict:
    """Check the GLOBAL persona-skill level distribution — the scoring label.

    This is distinct from check_proficiency_distribution, which measures the
    per-document evidence *intensity* grain. That grain looked balanced even when
    the global labels had collapsed (no level-1 skills), so it never caught the
    degenerate distribution. This check reads persona['skills'] directly — the
    exact (persona, skill) -> level pairs the scoring model trains on — and fails
    if any level falls below an 8% floor.
    """
    level_counts = Counter()
    for persona in personas:
        for _skill, level in persona.get("skills", {}).items():
            level_counts[int(level)] += 1

    total = sum(level_counts.values())
    distribution = {}
    for level in range(1, 6):
        count = level_counts.get(level, 0)
        pct = (count / total * 100) if total > 0 else 0
        distribution[level] = {
            "count": count,
            "percentage": round(pct, 1),
            "passed": pct >= 8,
        }

    return {
        "total_skills": total,
        "distribution": distribution,
        "all_passed": all(d["passed"] for d in distribution.values()),
        "note": "Global persona-skill level = scoring-model label; floor is 8% per level.",
    }


def check_archetype_distribution(personas: list[dict]) -> dict:
    """Check that all archetypes have >= 4% representation."""
    archetype_counts = Counter(
        p["hyperparams"]["archetype"] for p in personas
    )
    total = len(personas)

    distribution = {}
    for arch, count in sorted(archetype_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        distribution[arch] = {
            "count": count,
            "percentage": round(pct, 1),
            "passed": pct >= 4,
        }

    return {
        "total_personas": total,
        "distribution": distribution,
        "all_passed": all(d["passed"] for d in distribution.values()),
    }


def check_experience_distribution(personas: list[dict]) -> dict:
    """Check that experience distribution matches target (+-5%)."""
    targets = {
        "0-1": 15, "2-3": 20, "4-6": 25,
        "7-9": 20, "10-12": 12, "13+": 8,
    }

    bracket_counts = Counter()
    for p in personas:
        years = p["hyperparams"]["years_experience"]
        if years <= 1:
            bracket_counts["0-1"] += 1
        elif years <= 3:
            bracket_counts["2-3"] += 1
        elif years <= 6:
            bracket_counts["4-6"] += 1
        elif years <= 9:
            bracket_counts["7-9"] += 1
        elif years <= 12:
            bracket_counts["10-12"] += 1
        else:
            bracket_counts["13+"] += 1

    total = len(personas)
    distribution = {}
    for bracket, target_pct in targets.items():
        count = bracket_counts.get(bracket, 0)
        pct = (count / total * 100) if total > 0 else 0
        deviation = abs(pct - target_pct)
        distribution[bracket] = {
            "count": count,
            "percentage": round(pct, 1),
            "target": target_pct,
            "deviation": round(deviation, 1),
            "passed": deviation <= 5,
        }

    return {
        "total_personas": total,
        "distribution": distribution,
        "all_passed": all(d["passed"] for d in distribution.values()),
    }


def check_doc_type_distribution(documents_db: dict) -> dict:
    """Check document type distribution."""
    type_counts = Counter(doc["type"] for doc in documents_db.values())
    total = len(documents_db)

    distribution = {}
    for dt, count in sorted(type_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        distribution[dt] = {
            "count": count,
            "percentage": round(pct, 1),
        }

    return {
        "total_documents": total,
        "distribution": distribution,
    }


def check_length_proficiency_correlation(personas: list[dict], documents_db: dict) -> dict:
    """Check that document length doesn't correlate strongly with proficiency.

    Target: |Spearman r| < 0.3
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        logger.warning("scipy not installed — skipping correlation check")
        return {"skipped": True, "reason": "scipy not installed"}

    lengths = []
    levels = []

    for persona in personas:
        for doc_id in persona.get("document_ids", []):
            doc = documents_db.get(doc_id)
            if not doc:
                continue
            for skill, intensity in doc.get("skill_evidence", {}).items():
                lengths.append(len(doc.get("text", "")))
                levels.append(intensity)

    if len(lengths) < 10:
        return {"skipped": True, "reason": f"Too few examples ({len(lengths)})"}

    corr, p_val = spearmanr(lengths, levels)

    return {
        "spearman_r": round(float(corr), 4),
        "p_value": round(float(p_val), 6),
        "threshold": 0.3,
        "passed": bool(abs(float(corr)) < 0.3),
        "total_pairs": len(lengths),
    }


def run_stats_checks(
    personas: list[dict],
    documents_db: dict,
    output_dir: Path,
) -> dict:
    """Run all statistical checks and write report.

    Output: output_dir/stats_report.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "proficiency_distribution": check_proficiency_distribution(personas, documents_db),
        "global_level_distribution": check_global_level_distribution(personas),
        "archetype_distribution": check_archetype_distribution(personas),
        "experience_distribution": check_experience_distribution(personas),
        "doc_type_distribution": check_doc_type_distribution(documents_db),
        "length_proficiency_correlation": check_length_proficiency_correlation(personas, documents_db),
    }

    report_file = output_dir / "stats_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote stats report: {report_file}")

    return report

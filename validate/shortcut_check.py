"""
Validation: Anti-shortcut checks (BoW baseline, banned phrases).
All checks are read-only — output reports only.
"""

import re
import json
import logging
from pathlib import Path
from collections import Counter

from generate.sanitizer import COMPILED_PATTERNS

logger = logging.getLogger(__name__)


def check_banned_phrases(documents_db: dict) -> dict:
    """Scan all documents for banned proficiency keywords.

    Returns report dict.
    """
    issues = []
    for doc_id, doc in documents_db.items():
        text = doc.get("text", "")
        for pattern in COMPILED_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                issues.append({
                    "doc_id": doc_id,
                    "pattern": pattern.pattern,
                    "matches": matches,
                })

    return {
        "total_documents": len(documents_db),
        "documents_with_issues": len(set(i["doc_id"] for i in issues)),
        "total_issues": len(issues),
        "issues": issues,
    }


def check_level1_absence(personas: list[dict], documents_db: dict) -> dict:
    """Verify that level-1 skills never appear in any document text.

    Returns report dict.
    """
    issues = []

    for persona in personas:
        level1_skills = [
            s for s, level in persona.get("skills", {}).items() if level == 1
        ]

        for doc_id in persona.get("document_ids", []):
            doc = documents_db.get(doc_id)
            if not doc:
                continue

            text = doc.get("text", "").lower()
            for skill in level1_skills:
                # Check primary skill name
                skill_lower = skill.lower().split("/")[0].split("(")[0].strip()
                if len(skill_lower) > 2 and skill_lower in text:
                    issues.append({
                        "persona_id": persona["persona_id"],
                        "doc_id": doc_id,
                        "skill": skill,
                        "found_text": skill_lower,
                    })

    return {
        "total_checked": sum(
            len([s for s, l in p.get("skills", {}).items() if l == 1])
            * len(p.get("document_ids", []))
            for p in personas
        ),
        "violations": len(issues),
        "issues": issues,
    }


def check_bow_baseline(personas: list[dict], documents_db: dict) -> dict:
    """Train a BoW baseline and check if accuracy is suspiciously high.

    If TF-IDF + LogReg accuracy > 45%, there may be lexical shortcuts.
    Returns report dict.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError:
        logger.warning("scikit-learn not installed — skipping BoW baseline check")
        return {"skipped": True, "reason": "scikit-learn not installed"}

    # Build (skill + doc_text, level) pairs
    texts = []
    labels = []

    for persona in personas:
        for doc_id in persona.get("document_ids", []):
            doc = documents_db.get(doc_id)
            if not doc:
                continue

            skill_evidence = doc.get("skill_evidence", {})
            for skill, intensity in skill_evidence.items():
                texts.append(f"{skill} [SEP] {doc['text'][:1000]}")
                labels.append(intensity)

    if len(texts) < 50:
        return {"skipped": True, "reason": f"Too few examples ({len(texts)})"}

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=min(5, len(texts) // 10), scoring="accuracy")
    mean_acc = float(scores.mean())

    # Get top features per class for analysis
    clf.fit(X, y)
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if len(clf.classes_) > 2 else clf.coef_[0]
        top_idx = coefs.argsort()[-10:][::-1]
        top_features[str(cls)] = [
            {"feature": feature_names[idx], "weight": float(coefs[idx])}
            for idx in top_idx
        ]

    return {
        "mean_cv_accuracy": mean_acc,
        "cv_scores": [float(s) for s in scores],
        "threshold": 0.45,
        "passed": mean_acc < 0.45,
        "total_examples": len(texts),
        "label_distribution": dict(Counter(labels)),
        "top_features_per_class": top_features,
    }


def run_shortcut_checks(
    personas: list[dict],
    documents_db: dict,
    output_dir: Path,
) -> dict:
    """Run all anti-shortcut checks and write report.

    Output: output_dir/shortcut_check_report.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "banned_phrases": check_banned_phrases(documents_db),
        "level1_absence": check_level1_absence(personas, documents_db),
        "bow_baseline": check_bow_baseline(personas, documents_db),
    }

    report_file = output_dir / "shortcut_check_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote shortcut check report: {report_file}")

    return report

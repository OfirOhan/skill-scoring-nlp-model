"""
Phase 6: Anti-shortcut sanitization for generated documents.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Proficiency keywords that should NEVER appear in generated documents
BANNED_PATTERNS = [
    r"\bexpert\s+in\b",
    r"\bexpert\s+at\b",
    r"\bexpert\s+with\b",
    r"\bbeginner\s+at\b",
    r"\bbeginner\s+in\b",
    r"\bproficient\s+in\b",
    r"\bproficient\s+with\b",
    r"\badvanced\s+knowledge\b",
    r"\badvanced\s+skills?\b",
    r"\bbasic\s+understanding\b",
    r"\bbasic\s+knowledge\b",
    r"\bmastery\s+of\b",
    r"\bnovice\s+in\b",
    r"\bnovice\s+at\b",
    r"\bextensive\s+experience\s+with\b",
    r"\bextensive\s+experience\s+in\b",
    # Explicit proficiency ratings
    r"\b\w+\s*:\s*(?:expert|advanced|intermediate|beginner|basic|novice)\b",
    # Self-assessment labels
    r"\bhighly\s+proficient\b",
    r"\bdeeply\s+experienced\b",
    r"\bworld-class\b",
    r"\btop-tier\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BANNED_PATTERNS]


def sanitize_document(text: str, skill_evidence: dict) -> list[str]:
    """Check a generated document for proficiency keyword leakage.

    Args:
        text: The generated document text.
        skill_evidence: dict mapping skill -> local intensity (1-5).

    Returns:
        List of issue descriptions. Empty list means the document is clean.
    """
    issues = []

    # Check for banned proficiency keywords
    for pattern in COMPILED_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            issues.append(f"Banned pattern found: {matches}")

    # Check that level-1 skills don't appear in the text
    for skill, intensity in skill_evidence.items():
        if intensity == 1:
            # Check if the skill name (or its root) appears in the text
            # Use word boundary matching for skill names
            skill_words = skill.lower().split("/")[0].split("(")[0].strip()
            # Only check the primary name, not aliases
            if len(skill_words) > 2:  # Skip very short names that cause false positives
                escaped = re.escape(skill_words)
                if re.search(rf"\b{escaped}\b", text, re.IGNORECASE):
                    issues.append(
                        f"Level-1 skill '{skill}' found in document text"
                    )

    return issues

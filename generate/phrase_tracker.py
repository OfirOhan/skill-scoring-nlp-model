"""
Adaptive phrase tracking for document generation.

Tracks n-gram frequencies across generated documents and identifies
overused LLM-isms using TF-IDF-style logic. Phrases that are common
in general English (e.g., 'worked with', 'was responsible for') are
not penalized, while LLM-specific patterns (e.g., 'rare blend of',
'instrumental in shaping') are flagged.

Usage:
    tracker = PhraseTracker()
    # After each document is generated:
    tracker.ingest(doc_text)
    # Before generating the next document:
    banned = tracker.get_banned_phrases(top_k=10)
"""

import re
import math
from collections import Counter
from typing import Optional


# Baseline frequencies of common English n-grams (4-grams).
# These are naturally frequent in professional writing and should NOT
# be flagged as LLM-isms. Values are relative frequency scores (higher = more common).
BASELINE_COMMON_NGRAMS = {
    # Common professional/CV phrases
    "worked with the team": 5.0,
    "was responsible for the": 5.0,
    "as part of the": 5.0,
    "in order to improve": 4.0,
    "in collaboration with the": 4.0,
    "the development of the": 4.0,
    "as a member of": 4.0,
    "with a focus on": 4.0,
    "in the development of": 4.0,
    "contributed to the development": 4.0,
    "i am pleased to": 4.0,
    "i have had the": 4.0,
    "in my role as": 4.0,
    "over the past few": 3.5,
    "is a key part": 3.5,
    "in the context of": 3.5,
    "with more than years": 3.5,
    "years of experience in": 4.5,
    "of experience in the": 4.0,
    "experience in the field": 3.5,
    "to ensure that the": 3.5,
    "responsible for the design": 3.0,
    "in addition to the": 3.0,
    "as well as the": 3.5,
    # Common transitional phrases
    "on top of that": 3.0,
    "in addition to that": 3.0,
    "in the process of": 3.0,
    "at the same time": 3.0,
}

# Known LLM-ism seed phrases — these are always suspicious regardless
# of frequency. They get a baseline score of 0 (never common in real writing).
LLM_ISM_SEEDS = [
    "rare blend of technical",
    "rare blend of strategic",
    "a rare blend of",
    "standout engineer whose",
    "standout developer whose",
    "instrumental in shaping the",
    "instrumental in driving the",
    "transformative impact on the",
    "exemplifies a rare blend",
    "exemplify the rare blend",
    "whose technical rigor and",
    "technical precision and collaborative",
    "technical precision and strategic",
    "consistently delivering transformative",
    "consistently demonstrated exceptional",
    "passionate about building scalable",
    "passionate about leveraging technology",
    "passionate about driving innovation",
    "seamless integration of cloud",
    "seamlessly integrated with existing",
    "cutting edge solutions that",
    "spearheaded the modernization of",
    "spearheaded the migration of",
    "leveraging modern technologies to",
    "architecting scalable solutions that",
    "delivering measurable impact across",
    "measurable value across the",
    "blend of technical mastery",
    "thrive at the intersection",
    "at the intersection of",
]


class PhraseTracker:
    """Tracks n-gram frequencies across documents and identifies overused phrases.
    
    Uses TF-IDF-style scoring: phrases that are common in general English
    (high baseline frequency) are not penalized, while phrases that spike
    in the generated corpus but are rare in natural writing are flagged.
    
    Thread-safe for asyncio (single-threaded event loop, no mutex needed).
    """

    def __init__(self, ngram_sizes: tuple[int, ...] = (4, 5, 6)):
        self._ngram_counts: Counter = Counter()
        self._total_docs: int = 0
        self._ngram_sizes = ngram_sizes

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for n-gram extraction."""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract n-grams of configured sizes from text."""
        normalized = self._normalize(text)
        words = normalized.split()
        ngrams = []
        for n in self._ngram_sizes:
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngrams.append(ngram)
        return ngrams

    def ingest(self, document_text: str) -> None:
        """Ingest a generated document and update n-gram frequencies.
        
        Call this after each successful document generation.
        """
        ngrams = self._extract_ngrams(document_text)
        # Count unique ngrams per document (document frequency, not term frequency)
        unique_ngrams = set(ngrams)
        self._ngram_counts.update(unique_ngrams)
        self._total_docs += 1

    def _score_ngram(self, ngram: str, doc_freq: int) -> float:
        """Score an n-gram using TF-IDF-style logic.
        
        Returns a suspicion score. Higher = more likely an LLM-ism.
        
        Score = doc_frequency_ratio / baseline_frequency
        - High doc frequency + low baseline = LLM-ism (high score)
        - High doc frequency + high baseline = common English (low score)
        """
        if self._total_docs < 3:
            return 0.0

        # Document frequency ratio (what fraction of docs contain this phrase)
        df_ratio = doc_freq / self._total_docs

        # Baseline frequency (how common this is in general English)
        baseline = BASELINE_COMMON_NGRAMS.get(ngram, 0.0)

        # Known LLM-isms get baseline 0
        if ngram in LLM_ISM_SEEDS or any(ngram.startswith(seed[:20]) for seed in LLM_ISM_SEEDS):
            baseline = 0.0

        # Score: high doc frequency + low baseline = suspicious
        # Add 0.1 to baseline to avoid division by zero
        score = df_ratio / (baseline + 0.1)

        return score

    def get_banned_phrases(
        self,
        top_k: int = 10,
        min_doc_freq: int = 2,
        min_score: float = 0.3,
    ) -> list[str]:
        """Get the top-K most overused phrases to ban in the next generation.
        
        Args:
            top_k: Maximum number of phrases to return.
            min_doc_freq: Minimum number of documents a phrase must appear in.
            min_score: Minimum suspicion score to be considered.
            
        Returns:
            List of phrase strings to inject into the prompt as banned phrases.
        """
        if self._total_docs < 3:
            # Not enough data — return seed LLM-isms instead
            return LLM_ISM_SEEDS[:top_k]

        scored = []
        for ngram, count in self._ngram_counts.items():
            if count < min_doc_freq:
                continue
            score = self._score_ngram(ngram, count)
            if score >= min_score:
                scored.append((ngram, score, count))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [phrase for phrase, score, count in scored[:top_k]]

    def format_for_prompt(self, top_k: int = 10) -> str:
        """Format banned phrases as a prompt-ready string.
        
        Returns:
            A string like:
            'Avoid these overused phrases: "rare blend of", "instrumental in", ...'
            Or empty string if no phrases to ban.
        """
        phrases = self.get_banned_phrases(top_k=top_k)
        if not phrases:
            return ""

        quoted = [f'"{p}"' for p in phrases]
        return "Avoid these overused phrases (and close variants): " + ", ".join(quoted)

    @property
    def total_docs(self) -> int:
        return self._total_docs

    @property
    def top_ngrams(self) -> list[tuple[str, int]]:
        """Return most common n-grams for debugging."""
        return self._ngram_counts.most_common(30)

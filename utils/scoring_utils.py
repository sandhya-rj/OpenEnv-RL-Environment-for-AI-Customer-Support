"""
utils/scoring_utils.py — Keyword matching and statistical similarity utilities.

Design rules
------------
• All functions are pure and deterministic
• Return values are documented with exact ranges
• No imports from other project modules (zero dependency)
• Edge cases are handled gracefully (empty sets, zero denominators)

Public API
----------
keyword_hits(norm_text, keywords)               -> int    count of matching keywords
keyword_ratio(norm_text, keywords)              -> float  [0, 1]
overlap_ratio(text_a, text_b, stop_words)       -> float  [0, 1]
jaccard_similarity(set_a, set_b)                -> float  [0, 1]
char_jaccard(text_a, text_b, n)                 -> float  [0, 1]
diminishing_sum(hits, base, decay)              -> float  ≥ 0 (caller must clamp)
sigmoid_score(x, midpoint, steepness)           -> float  [0, 1]
"""

from __future__ import annotations

import math
from typing import FrozenSet, List, Set

from utils.text_processing import char_ngrams, content_tokens

__all__ = [
    "keyword_hits",
    "keyword_ratio",
    "jaccard_similarity",
    "overlap_ratio",
    "char_jaccard",
    "diminishing_sum",
    "sigmoid_score",
]


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def keyword_hits(norm_text: str, keywords: List[str]) -> int:
    """
    Count how many distinct keywords appear as substrings in ``norm_text``.

    Substring (not whole-word) matching handles morphological variants:
    e.g. "reimburse" matches both "reimburse" and "reimbursement".

    Parameters
    ----------
    norm_text : str   Pre-normalised text (output of normalize()).
    keywords  : List[str]

    Returns
    -------
    int in [0, len(keywords)]
    """
    if not norm_text:
        return 0
    return sum(1 for kw in keywords if kw in norm_text)


def keyword_ratio(norm_text: str, keywords: List[str]) -> float:
    """
    Fraction of ``keywords`` found in ``norm_text``.

    Returns 0.0 when ``keywords`` is empty.
    """
    if not keywords:
        return 0.0
    return keyword_hits(norm_text, keywords) / len(keywords)


# ---------------------------------------------------------------------------
# Set-overlap scoring
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|.

    Returns 0.0 when both sets are empty.
    """
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def overlap_ratio(
    text_a: str,
    text_b: str,
    stop_words: FrozenSet[str],
    min_length: int = 3,
) -> float:
    """
    Content-word overlap of ``text_a`` relative to ``text_b``.

    Computed as: |content_words(a) ∩ content_words(b)| / |content_words(b)|

    Returns 0.0 when ``text_b`` has no qualifying content words.
    """
    words_b = content_tokens(text_b, stop_words, min_length)
    if not words_b:
        return 0.0
    words_a = content_tokens(text_a, stop_words, min_length)
    return len(words_a & words_b) / len(words_b)


# ---------------------------------------------------------------------------
# Character n-gram Jaccard similarity  (repetition detection)
# ---------------------------------------------------------------------------

def char_jaccard(text_a: str, text_b: str, n: int = 4) -> float:
    """
    Jaccard similarity between character n-gram sets of two texts.

    Returns 0.0 when both texts normalise to empty.
    """
    return jaccard_similarity(char_ngrams(text_a, n), char_ngrams(text_b, n))


# ---------------------------------------------------------------------------
# Score shaping utilities
# ---------------------------------------------------------------------------

def diminishing_sum(hits: int, base: float = 0.10, decay: float = 0.82) -> float:
    """
    Geometric series with decay:  Σ base * decay^i,  for i in [0, hits).

    Models diminishing returns — each additional keyword hit contributes less
    than the previous one. Returns 0.0 when hits == 0.

    The caller is responsible for clamping the result to [0, 1].
    """
    if hits <= 0:
        return 0.0
    return sum(base * (decay ** i) for i in range(hits))


def sigmoid_score(
    x: float,
    midpoint: float = 0.3,
    steepness: float = 8.0,
) -> float:
    """
    Smooth logistic function mapping x ∈ [0, 1] → [0, 1].

    • At x == midpoint  → 0.5
    • Larger steepness  → sharper S-curve near midpoint

    Returns a value in [0.0, 1.0], handling overflow gracefully.
    """
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0

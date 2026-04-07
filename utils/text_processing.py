"""
utils/text_processing.py — Pure, deterministic text normalisation utilities.

Design rules
------------
• No state — every function is a pure function (same input → same output, always)
• No imports from other project modules (zero dependency)
• All functions are typed and documented
• Graceful degradation on edge cases: empty strings return safe defaults

Public API
----------
remove_punctuation(text)                        -> str
normalize(text)                                 -> str
tokenize(text)                                  -> List[str]
content_tokens(text, stop_words, min_length)    -> Set[str]
char_ngrams(text, n)                            -> Set[str]
word_count(text)                                -> int
expand_with_synonyms(keywords, synonyms)        -> List[str]
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Set

__all__ = [
    "remove_punctuation",
    "normalize",
    "tokenize",
    "content_tokens",
    "char_ngrams",
    "word_count",
    "expand_with_synonyms",
]


# ---------------------------------------------------------------------------
# Core normalisation pipeline
# ---------------------------------------------------------------------------

def remove_punctuation(text: str) -> str:
    """Replace all non-alphanumeric, non-whitespace characters with a space."""
    return re.sub(r"[^\w\s]", " ", text)


def normalize(text: str) -> str:
    """
    Canonical text normalisation:
      1. Lowercase
      2. Replace punctuation with spaces
      3. Collapse consecutive whitespace

    Returns a stripped, single-spaced, all-lowercase string.
    Returns "" for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Normalise then split into individual word tokens.

    Returns an empty list for empty or whitespace-only input.
    """
    norm = normalize(text)
    return norm.split() if norm else []


def word_count(text: str) -> int:
    """Return the number of whitespace-delimited tokens in the raw text."""
    return len(text.split()) if text and text.strip() else 0


# ---------------------------------------------------------------------------
# Content-word extraction
# ---------------------------------------------------------------------------

def content_tokens(
    text: str,
    stop_words: FrozenSet[str],
    min_length: int = 3,
) -> Set[str]:
    """
    Extract unique content words from normalised text.

    A content word must:
      1. NOT be in stop_words
      2. Have length >= min_length (default: 3)

    Returns an empty set for empty input or when all words are stop words.
    """
    return {
        token
        for token in tokenize(text)
        if token not in stop_words and len(token) >= min_length
    }


# ---------------------------------------------------------------------------
# Character n-gram extraction  (used for repetition detection)
# ---------------------------------------------------------------------------

def char_ngrams(text: str, n: int = 4) -> Set[str]:
    """
    Return the set of character n-grams of the normalised text.

    Edge cases:
    - Empty input         → empty set
    - Shorter than n      → single-element set containing the full normalised string
    - Normal input        → full sliding-window n-gram set

    Parameters
    ----------
    text : str
    n    : int   N-gram size (default 4).
    """
    norm = normalize(text)
    if not norm:
        return set()
    if len(norm) < n:
        return {norm}
    return {norm[i: i + n] for i in range(len(norm) - n + 1)}


# ---------------------------------------------------------------------------
# Synonym expansion
# ---------------------------------------------------------------------------

def expand_with_synonyms(
    keywords: List[str],
    synonyms: Dict[str, List[str]],
) -> List[str]:
    """
    Return a deduplicated union of keywords and their known synonyms.

    Order is preserved: original keywords first, then synonyms in order.

    Parameters
    ----------
    keywords : List[str]
    synonyms : Dict mapping keyword -> list of synonym strings
    """
    seen: set[str]  = set()
    result: List[str] = []
    for item in keywords + [s for kw in keywords for s in synonyms.get(kw, [])]:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

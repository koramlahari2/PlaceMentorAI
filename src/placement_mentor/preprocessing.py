"""Lightweight, deterministic text normalization for FAQ retrieval."""

import re
import unicodedata


def preprocess_text(text: str) -> str:
    """Normalize Unicode, casing, punctuation, and whitespace without stemming."""
    if not isinstance(text, str):
        raise TypeError("Text to preprocess must be a string.")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"_+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()

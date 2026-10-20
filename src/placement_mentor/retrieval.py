"""Pretrained sentence-embedding based FAQ retrieval."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from placement_mentor.preprocessing import preprocess_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.35
FALLBACK_RESPONSE = (
    "I couldn't find a sufficiently relevant answer in the placement FAQ. "
    "Try rephrasing your question or choosing a different category."
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAQ_PATH = PROJECT_ROOT / "data" / "placement_faq.csv"
REQUIRED_COLUMNS = ("id", "category", "question", "answer")


@dataclass(frozen=True)
class FAQMatch:
    """A retrieved FAQ or a fallback when the best score is below threshold."""

    matched: bool
    faq_id: str | None
    category: str | None
    question: str | None
    answer: str
    similarity_score: float


@lru_cache(maxsize=2)
def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and cache the pretrained model on CPU for local inference."""
    return SentenceTransformer(model_name, device="cpu")


@lru_cache(maxsize=8)
def _read_faqs(path: str, _modified_ns: int) -> pd.DataFrame:
    if Path(path).stat().st_mtime_ns != _modified_ns:
        raise RuntimeError("FAQ CSV changed while it was being loaded; retry.")

    faqs = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = set(REQUIRED_COLUMNS).difference(faqs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"FAQ CSV is missing required columns: {missing}")

    faqs = faqs.loc[:, list(REQUIRED_COLUMNS)].copy()
    if faqs.empty:
        raise ValueError("FAQ CSV contains no FAQ rows.")
    if faqs.loc[:, list(REQUIRED_COLUMNS)].apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError("FAQ CSV contains an empty required field.")
    if faqs["id"].duplicated().any():
        raise ValueError("FAQ CSV contains duplicate IDs.")
    if faqs["question"].str.casefold().duplicated().any():
        raise ValueError("FAQ CSV contains duplicate questions.")
    return faqs.reset_index(drop=True)


def load_faqs(faq_path: str | Path = DEFAULT_FAQ_PATH) -> pd.DataFrame:
    """Load and validate the FAQ CSV, returning a copy of its cached table."""
    path = Path(faq_path).resolve()
    modified_ns = path.stat().st_mtime_ns
    return _read_faqs(str(path), modified_ns).copy()


@lru_cache(maxsize=8)
def _embed_faqs(path: str, modified_ns: int, model_name: str) -> np.ndarray:
    faqs = _read_faqs(path, modified_ns)
    model = load_model(model_name)
    questions = [preprocess_text(question) for question in faqs["question"]]
    embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def generate_faq_embeddings(
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    model_name: str = MODEL_NAME,
) -> np.ndarray:
    """Generate or retrieve cached embeddings for every FAQ question."""
    path = Path(faq_path).resolve()
    modified_ns = path.stat().st_mtime_ns
    return _embed_faqs(str(path), modified_ns, model_name).copy()


def find_best_match(
    question: str,
    category: str | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    model_name: str = MODEL_NAME,
) -> FAQMatch:
    """Find the highest-scoring FAQ and apply a configurable cosine threshold."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold must be between -1.0 and 1.0.")

    processed_question = preprocess_text(question)
    if not processed_question:
        return FAQMatch(
            matched=False,
            faq_id=None,
            category=None,
            question=None,
            answer=FALLBACK_RESPONSE,
            similarity_score=0.0,
        )

    path = Path(faq_path).resolve()
    modified_ns = path.stat().st_mtime_ns
    faqs = _read_faqs(str(path), modified_ns)
    faq_embeddings = _embed_faqs(str(path), modified_ns, model_name)

    if category is None:
        candidate_indices = np.arange(len(faqs))
    else:
        candidate_indices = np.flatnonzero(faqs["category"].eq(category).to_numpy())
        if candidate_indices.size == 0:
            raise ValueError(f"No FAQs found for category: {category}")

    query_embedding = load_model(model_name).encode(
        [processed_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = cosine_similarity(
        np.asarray(query_embedding, dtype=np.float32),
        faq_embeddings[candidate_indices],
    )[0]
    best_position = int(np.argmax(scores))
    best_index = int(candidate_indices[best_position])
    best_score = float(scores[best_position])
    best_faq = faqs.iloc[best_index]

    if best_score < threshold:
        return FAQMatch(
            matched=False,
            faq_id=None,
            category=None,
            question=None,
            answer=FALLBACK_RESPONSE,
            similarity_score=best_score,
        )

    return FAQMatch(
        matched=True,
        faq_id=best_faq["id"],
        category=best_faq["category"],
        question=best_faq["question"],
        answer=best_faq["answer"],
        similarity_score=best_score,
    )

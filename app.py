"""Streamlit chat interface for the PlaceMentor AI FAQ retriever."""

from typing import TypedDict

import streamlit as st
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from requests.exceptions import RequestException

from placement_mentor.retrieval import (
    DEFAULT_FAQ_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    FAQMatch,
    find_best_match,
    load_faqs,
)


class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    similarity_score: float
    faq_question: str
    category: str
    error: bool


st.set_page_config(
    page_title="PlaceMentor AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --pm-ink: #17243a;
        --pm-muted: #62718a;
        --pm-accent: #2764e7;
        --pm-border: #e2e8f0;
        --pm-surface: #f6f8fc;
    }
    .stApp {
        background: linear-gradient(180deg, #f7f9fd 0%, #ffffff 34rem);
        color: var(--pm-ink);
    }
    [data-testid="stSidebar"] {
        background: #f4f7fc;
        border-right: 1px solid var(--pm-border);
    }
    .block-container {
        max-width: 1050px;
        padding-top: 2.2rem;
        padding-bottom: 7rem;
    }
    .pm-eyebrow {
        color: var(--pm-accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .pm-subtitle {
        color: var(--pm-muted);
        font-size: 1.05rem;
        margin-top: -0.35rem;
        max-width: 48rem;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--pm-border);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
    }
    [data-testid="stChatInput"] {
        border-color: var(--pm-border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def store_answer(question: str, category: str | None, threshold: float) -> None:
    """Retrieve one answer and append the user/assistant turns to session state."""
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Searching the placement FAQ..."):
            result: FAQMatch = find_best_match(
                question,
                category=category,
                threshold=threshold,
            )
    except (
        OSError,
        RuntimeError,
        ValueError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RequestException,
    ) as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"FAQ search could not be completed: {error}",
                "error": True,
            }
        )
        return

    message: ChatMessage = {
        "role": "assistant",
        "content": result.answer,
        "similarity_score": result.similarity_score,
    }
    if result.matched:
        message["faq_question"] = result.question or ""
        message["category"] = result.category or ""
    st.session_state.messages.append(message)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.sidebar.markdown("## PlaceMentor AI")
st.sidebar.caption("Placement preparation, grounded in a curated FAQ.")
faq_data = load_faqs(DEFAULT_FAQ_PATH)
categories = sorted(faq_data["category"].unique().tolist())
selected_category = st.sidebar.selectbox(
    "Placement category",
    options=["All categories", *categories],
    index=0,
)
active_category = None if selected_category == "All categories" else selected_category
similarity_threshold = st.sidebar.slider(
    "Semantic similarity threshold",
    min_value=0.20,
    max_value=0.80,
    value=DEFAULT_SIMILARITY_THRESHOLD,
    step=0.01,
    help=(
        "A candidate below this cosine-similarity score receives the fallback "
        "instead of an FAQ answer. This score is not a probability."
    ),
)
st.sidebar.caption("Model: all-MiniLM-L6-v2 · Local CPU inference")

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.rerun()

st.markdown('<div class="pm-eyebrow">Campus placement preparation</div>', unsafe_allow_html=True)
st.title("Ask a placement question")
st.markdown(
    '<p class="pm-subtitle">Get a concise answer retrieved from the placement '
    "FAQ across aptitude, coding, interviews, resumes, and the placement process."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("#### Suggested questions")
suggestion_rows = faq_data
if active_category is not None:
    suggestion_rows = faq_data.loc[faq_data["category"].eq(active_category)]
suggestions = suggestion_rows["question"].head(3).tolist()
suggestion_columns = st.columns(len(suggestions))
for index, (column, suggestion) in enumerate(zip(suggestion_columns, suggestions)):
    if column.button(suggestion, key=f"suggestion-{index}", use_container_width=True):
        st.session_state.pending_question = suggestion

for chat_message in st.session_state.messages:
    with st.chat_message(chat_message["role"]):
        if chat_message.get("error"):
            st.error(chat_message["content"])
        else:
            st.markdown(chat_message["content"])
            if chat_message["role"] == "assistant":
                if chat_message.get("faq_question"):
                    st.caption(
                        f"Matched FAQ: {chat_message['faq_question']} · "
                        f"Category: {chat_message['category']}"
                    )
                st.caption(
                    "Semantic similarity score: "
                    f"{chat_message.get('similarity_score', 0.0):.3f}"
                )

typed_question = st.chat_input("Ask about placements, interviews, coding, or resumes")
question_to_answer = typed_question or st.session_state.pending_question
st.session_state.pending_question = None
if question_to_answer and question_to_answer.strip():
    store_answer(question_to_answer.strip(), active_category, similarity_threshold)
    st.rerun()

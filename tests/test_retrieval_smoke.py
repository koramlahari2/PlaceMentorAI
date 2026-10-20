"""Four real-model smoke checks for semantic FAQ matching."""

from placement_mentor.retrieval import (
    DEFAULT_FAQ_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    FALLBACK_RESPONSE,
    find_best_match,
    generate_faq_embeddings,
    load_faqs,
)


def main() -> None:
    faqs = load_faqs(DEFAULT_FAQ_PATH)
    embeddings = generate_faq_embeddings(DEFAULT_FAQ_PATH)

    assert len(faqs) == 101, f"Expected 101 FAQs, got {len(faqs)}"
    assert embeddings.shape[0] == len(faqs), (
        f"Expected one embedding per FAQ, got shape {embeddings.shape}"
    )

    test_questions = (
        (
            "How can I get faster at aptitude tests without making careless mistakes?",
            "Aptitude & Reasoning",
        ),
        (
            "What search should I use to find the fewest edges between graph nodes?",
            "Coding & DSA",
        ),
        (
            "How can I describe my personal projects on a CV for my first job?",
            "Resume & LinkedIn",
        ),
        (
            "What do I need to prepare for an interview?",
            "HR Interviews",
        ),
    )

    for question, expected_category in test_questions:
        result = find_best_match(question, threshold=DEFAULT_SIMILARITY_THRESHOLD)
        assert result.matched, (
            f"No match for {question!r}; score={result.similarity_score:.3f}"
        )
        assert result.category == expected_category, (
            f"Expected {expected_category!r}, got {result.category!r}"
        )
        print(
            f"Question: {question}\n"
            f"FAQ: {result.question}\n"
            f"Category: {result.category}\n"
            f"Semantic similarity score: {result.similarity_score:.3f}\n"
        )

    filtered = find_best_match(
        test_questions[2][0],
        category="Resume & LinkedIn",
    )
    assert filtered.matched and filtered.category == "Resume & LinkedIn"

    strict_result = find_best_match(test_questions[0][0], threshold=0.99)
    assert not strict_result.matched
    assert strict_result.answer == FALLBACK_RESPONSE

    empty_result = find_best_match("   ")
    assert not empty_result.matched
    assert empty_result.similarity_score == 0.0

    unrelated_result = find_best_match("What is the best way to brew coffee?")
    assert not unrelated_result.matched
    assert unrelated_result.answer == FALLBACK_RESPONSE

    print(f"PASS: {len(test_questions)} questions matched; embeddings={embeddings.shape}")
    print("PASS: category filtering, threshold fallback, empty input, and unrelated input")


if __name__ == "__main__":
    main()

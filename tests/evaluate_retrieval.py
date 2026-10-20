"""Run held-out semantic matching examples and report measured retrieval scores."""

import csv
from pathlib import Path

from placement_mentor.retrieval import DEFAULT_SIMILARITY_THRESHOLD, find_best_match

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation.csv"


def main() -> None:
    with EVALUATION_PATH.open(encoding="utf-8-sig", newline="") as file:
        examples = list(csv.DictReader(file))

    correct = 0
    answerable = 0
    correct_retrievals = 0

    for example in examples:
        expected_match = example["expected_match"].strip().casefold() == "true"
        expected_faq_id = example["expected_faq_id"].strip()
        result = find_best_match(
            example["query"],
            category=example["category"].strip() or None,
            threshold=DEFAULT_SIMILARITY_THRESHOLD,
        )
        is_correct = result.matched == expected_match
        if expected_match:
            answerable += 1
            is_correct = is_correct and result.faq_id == expected_faq_id
            correct_retrievals += int(result.matched and result.faq_id == expected_faq_id)
        correct += int(is_correct)

        predicted = result.faq_id if result.matched else "FALLBACK"
        status = "PASS" if is_correct else "MISS"
        print(
            f"{status} | expected={expected_faq_id or 'FALLBACK':7} "
            f"predicted={predicted:7} score={result.similarity_score:.3f} "
            f"| {example['query']}"
        )

    accuracy = correct / len(examples) if examples else 0.0
    retrieval_rate = correct_retrievals / answerable if answerable else 0.0
    print(
        f"\nMeasured threshold: {DEFAULT_SIMILARITY_THRESHOLD:.2f}"
        f"\nCorrect examples: {correct}/{len(examples)}"
        f"\nOverall evaluation accuracy: {accuracy:.1%}"
        f"\nTop-1 retrieval accuracy on answerable questions: "
        f"{correct_retrievals}/{answerable} ({retrieval_rate:.1%})"
    )


if __name__ == "__main__":
    main()

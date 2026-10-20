"""Focused tests for semantic retrieval and the Streamlit chat experience."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from placement_mentor.preprocessing import preprocess_text
from placement_mentor.retrieval import (
    DEFAULT_FAQ_PATH,
    FALLBACK_RESPONSE,
    find_best_match,
    generate_faq_embeddings,
    load_faqs,
)
from streamlit.testing.v1 import AppTest


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.faqs = load_faqs(DEFAULT_FAQ_PATH)
        cls.embeddings = generate_faq_embeddings(DEFAULT_FAQ_PATH)

    def test_dataset_has_one_embedding_per_faq(self) -> None:
        self.assertEqual(len(self.faqs), 101)
        self.assertEqual(self.embeddings.shape, (len(self.faqs), 384))

    def test_exact_question_retrieves_its_faq(self) -> None:
        result = find_best_match(
            "Should I focus on speed or accuracy in aptitude?",
            category="Aptitude & Reasoning",
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.faq_id, "APT003")
        self.assertAlmostEqual(result.similarity_score, 1.0, places=5)

    def test_interview_preparation_is_distinct_from_questions_to_ask(self) -> None:
        preparation = find_best_match("What do I need to prepare for an interview?")
        self.assertTrue(preparation.matched)
        self.assertEqual(preparation.faq_id, "HR016")

        questions_to_ask = find_best_match(
            "What are good questions to ask a hiring manager at the end of an interview?"
        )
        self.assertTrue(questions_to_ask.matched)
        self.assertEqual(questions_to_ask.faq_id, "HR014")

    def test_text_preprocessing_is_repeatable(self) -> None:
        self.assertEqual(
            preprocess_text("  Résumé—Tips! "),
            preprocess_text("résumé tips"),
        )

    def test_empty_question_uses_fallback(self) -> None:
        result = find_best_match("   ")
        self.assertFalse(result.matched)
        self.assertEqual(result.answer, FALLBACK_RESPONSE)
        self.assertEqual(result.similarity_score, 0.0)

    def test_category_filter_and_invalid_category(self) -> None:
        result = find_best_match(
            "How do I describe a personal project on my CV?",
            category="Resume & LinkedIn",
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.category, "Resume & LinkedIn")
        with self.assertRaisesRegex(ValueError, "No FAQs found"):
            find_best_match("Any question", category="Unknown category")

    def test_configurable_threshold_returns_fallback(self) -> None:
        result = find_best_match(
            "How can I keep up the pace but avoid calculation mistakes?",
            threshold=0.99,
        )
        self.assertFalse(result.matched)
        self.assertEqual(result.answer, FALLBACK_RESPONSE)
        with self.assertRaisesRegex(ValueError, "threshold"):
            find_best_match("A question", threshold=1.1)

    def test_unrelated_question_returns_fallback(self) -> None:
        result = find_best_match("How do I make a sourdough starter?")
        self.assertFalse(result.matched)
        self.assertEqual(result.answer, FALLBACK_RESPONSE)


class ChatInterfaceTests(unittest.TestCase):
    def test_category_chat_history_and_clear(self) -> None:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90).run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])

        app.selectbox[0].select("Coding & DSA").run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertTrue(
            any("coding interviews" in button.label.casefold() for button in app.button)
        )

        app.chat_input[0].set_value(
            "What search finds the shortest unweighted graph path?"
        ).run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        messages = app.session_state["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["category"], "Coding & DSA")

        clear_button = next(button for button in app.button if button.label == "Clear chat")
        clear_button.click().run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertEqual(app.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main()

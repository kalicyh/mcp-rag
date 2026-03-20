from __future__ import annotations

import unittest

from mcp_rag.retrieval.query_classifier import QueryClassifier, QueryIntent


class QueryClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = QueryClassifier()

    def test_troubleshooting_query(self) -> None:
        result = self.classifier.classify("I am getting an error in FastAPI")
        self.assertEqual(result.primary_intent, QueryIntent.TROUBLESHOOTING)
        self.assertTrue(result.is_technical)
        self.assertGreater(result.confidence, 0.3)
        self.assertIn("fastapi", [keyword.lower() for keyword in result.keywords])

    def test_non_technical_query(self) -> None:
        result = self.classifier.classify("Hello, how are you?")
        self.assertFalse(result.is_technical)
        self.assertEqual(result.primary_intent, QueryIntent.GENERAL_QA)

    def test_how_to_query(self) -> None:
        result = self.classifier.classify("How to build a Python API")
        self.assertEqual(result.primary_intent, QueryIntent.HOW_TO)
        self.assertTrue(result.is_technical)
        self.assertGreater(result.confidence, 0.3)

    def test_keyword_extraction_limits_stop_words(self) -> None:
        result = self.classifier.classify("What are the best practices for FastAPI and Python?")
        self.assertIn("fastapi", [keyword.lower() for keyword in result.keywords])
        self.assertIn("python", [keyword.lower() for keyword in result.keywords])
        self.assertLessEqual(len(result.keywords), 10)


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from src.evaluation import (
    extract_numbers,
    load_gold_set,
    parse_judge_response,
    score_answer,
)


class EvaluationTests(unittest.TestCase):
    def test_extract_numbers_handles_grouping_signs_and_trailing_punctuation(self):
        self.assertEqual(
            extract_numbers("Customer 18102, revenue $349,164.35; returns -1,200.00"),
            [18102.0, 349164.35, -1200.0],
        )
        self.assertEqual(
            extract_numbers("€1.234.567,89 and $1,234,567.89"),
            [1234567.89, 1234567.89],
        )

    def test_score_answer_requires_all_numbers_and_text(self):
        gold = {
            "expected_numbers": [18102, 349164.35],
            "required_text": ["customer"],
            "absolute_tolerance": 0.01,
            "relative_tolerance": 0.000001,
        }
        self.assertTrue(
            score_answer("Customer 18102, revenue 349,164.35", gold)["passed"]
        )
        self.assertFalse(score_answer("Customer 18102", gold)["passed"])
        self.assertFalse(
            score_answer("Item 18102, revenue 349,164.35", gold)["passed"]
        )

    def test_score_answer_supports_text_only_answers(self):
        country_gold = {
            "expected_numbers": [],
            "required_text": ["united kingdom"],
            "absolute_tolerance": 0.01,
            "relative_tolerance": 0.000001,
        }
        customer_gold = {
            "expected_numbers": [18102],
            "required_text": [],
            "absolute_tolerance": 0,
            "relative_tolerance": 0,
        }
        self.assertTrue(score_answer("United Kingdom", country_gold)["passed"])
        self.assertFalse(score_answer("France", country_gold)["passed"])
        self.assertTrue(score_answer("18102", customer_gold)["passed"])

    def test_score_answer_matches_locale_formatted_numeric_answers(self):
        gold = {
            "expected_numbers": [1234567.89],
            "required_text": [],
            "absolute_tolerance": 0.01,
            "relative_tolerance": 0.000001,
        }
        self.assertTrue(
            score_answer("1.234.567,89", gold)["passed"]
        )
        self.assertTrue(
            score_answer("1,234,567.89", gold)["passed"]
        )

    def test_gold_set_has_unique_records(self):
        gold = load_gold_set()
        self.assertEqual(len(gold), 6)
        self.assertEqual(len({record["id"] for record in gold}), len(gold))

    def test_parse_judge_response_preserves_benchmark_order(self):
        expected_ids = ["first", "second"]
        response = json.dumps(
            {
                "results": [
                    {
                        "id": "second",
                        "correctness": False,
                        "faithfulness": True,
                        "rationale": "The number is incorrect.",
                    },
                    {
                        "id": "first",
                        "correctness": True,
                        "faithfulness": True,
                        "rationale": "It matches the target.",
                    },
                ]
            }
        )
        parsed = parse_judge_response(response, expected_ids)
        self.assertEqual([result["id"] for result in parsed], expected_ids)

    def test_parse_judge_response_rejects_missing_ids(self):
        response = json.dumps(
            {
                "results": [
                    {
                        "id": "first",
                        "correctness": True,
                        "faithfulness": True,
                        "rationale": "It matches the target.",
                    }
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            parse_judge_response(response, ["first", "second"])


if __name__ == "__main__":
    unittest.main()

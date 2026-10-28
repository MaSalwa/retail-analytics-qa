"""Helpers for scoring table-Q&A answers against a frozen gold set."""

from __future__ import annotations

import json
import re
from pathlib import Path

GOLD_PATH = Path(__file__).resolve().parents[1] / "eval" / "gold_generation.jsonl"
NUMBER_PATTERN = re.compile(
    r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?![\w])"
)


def load_gold_set(path: str | Path = GOLD_PATH) -> list[dict]:
    """Load and validate the JSONL benchmark."""
    gold_path = Path(path)
    if not gold_path.is_file():
        raise FileNotFoundError(f"Gold set not found: {gold_path}")

    records = []
    with gold_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {
                "id",
                "question",
                "expected_numbers",
                "required_text",
                "absolute_tolerance",
                "relative_tolerance",
                "notes",
                "expected_display",
            }
            missing = required.difference(record)
            if missing:
                raise ValueError(
                    f"{gold_path}:{line_number} is missing fields: {sorted(missing)}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"Gold set is empty: {gold_path}")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError(f"Gold set contains duplicate IDs: {gold_path}")
    return records


def extract_numbers(text: str) -> list[float]:
    """Extract numbers using the most likely comma/dot locale convention."""
    return [_parse_number_token(match)[0] for match in NUMBER_PATTERN.findall(text)]


def _parse_number_token(token: str) -> list[float]:
    sign = token[:1] if token[:1] in "+-" else ""
    value = token[len(sign):]
    separators = [index for index, char in enumerate(value) if char in ".,"]
    if not separators:
        return [float(token)]

    candidates = []
    decimal_index = separators[-1]
    decimal_separator = value[decimal_index]
    integer_part = value[:decimal_index]
    decimal_part = value[decimal_index + 1:]
    normalized_integer = integer_part.replace(".", "").replace(",", "")
    candidates.append(float(f"{sign}{normalized_integer}.{decimal_part}"))

    groups = re.split(r"[.,]", value)
    if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
        candidates.append(float(sign + "".join(groups)))

    return list(dict.fromkeys(candidates))


def score_answer(answer: str, gold: dict) -> dict:
    """Check all expected numeric values and required text in an answer."""
    actual_numbers = [
        candidate
        for token in NUMBER_PATTERN.findall(answer)
        for candidate in _parse_number_token(token)
    ]
    absolute_tolerance = float(gold["absolute_tolerance"])
    relative_tolerance = float(gold["relative_tolerance"])

    numeric_checks = []
    for expected in gold["expected_numbers"]:
        expected_value = float(expected)
        tolerance = max(
            absolute_tolerance,
            abs(expected_value) * relative_tolerance,
        )
        matched_value = next(
            (
                actual
                for actual in actual_numbers
                if abs(actual - expected_value) <= tolerance
            ),
            None,
        )
        numeric_checks.append(
            {
                "expected": expected_value,
                "matched": matched_value is not None,
                "actual_match": matched_value,
                "tolerance": tolerance,
            }
        )

    normalized_answer = answer.casefold()
    text_checks = [
        {
            "expected": expected,
            "matched": expected.casefold() in normalized_answer,
        }
        for expected in gold["required_text"]
    ]
    passed = all(check["matched"] for check in numeric_checks) and all(
        check["matched"] for check in text_checks
    )
    return {
        "numeric_checks": numeric_checks,
        "text_checks": text_checks,
        "passed": passed,
    }


def build_judge_prompt(results: list[dict]) -> str:
    """Build one compact prompt to judge correctness and faithfulness."""
    cases = []
    for result in results:
        gold = result["gold"]
        cases.append(
            {
                "id": gold["id"],
                "question": gold["question"],
                "expected_answer": gold["expected_display"],
                "ground_truth_notes": gold["notes"],
                "numeric_match": result["numeric_score"]["passed"],
                "system_answer": result["answer"],
            }
        )

    return (
        "Evaluate each table-Q&A answer independently using only the supplied "
        "question, expected answer, and ground-truth notes. `correctness` is "
        "true only when the answer gives the correct result for the question. "
        "`faithfulness` is true only when its claims are supported by the "
        "ground-truth notes and expected answer. A numeric_match=false is "
        "strong evidence of incorrectness; do not override it without a clear "
        "reason. Return only valid JSON with this shape: "
        '{"results":[{"id":"...","correctness":true,"faithfulness":true,'
        '"rationale":"one short sentence"}]}. Include every supplied ID once. '
        "Do not add markdown fences or other text.\n\n"
        f"Cases:\n{json.dumps(cases, ensure_ascii=False)}"
    )


def parse_judge_response(response: str, expected_ids: list[str]) -> list[dict]:
    """Parse and validate the judge's JSON response."""
    content = response.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)

    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("LLM judge response must be a JSON object.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("LLM judge response must contain a 'results' list.")

    by_id = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Each LLM judge result must be a JSON object.")
        record_id = result.get("id")
        if not isinstance(record_id, str):
            raise ValueError("Each LLM judge result must have a string ID.")
        if record_id in by_id:
            raise ValueError(f"LLM judge returned duplicate ID: {record_id}")
        if not isinstance(result.get("correctness"), bool):
            raise ValueError(f"LLM judge correctness must be boolean for {record_id}.")
        if not isinstance(result.get("faithfulness"), bool):
            raise ValueError(f"LLM judge faithfulness must be boolean for {record_id}.")
        if not isinstance(result.get("rationale"), str):
            raise ValueError(f"LLM judge rationale must be text for {record_id}.")
        by_id[record_id] = result

    expected_id_set = set(expected_ids)
    if set(by_id) != expected_id_set:
        missing = sorted(expected_id_set.difference(by_id))
        unexpected = sorted(set(by_id).difference(expected_id_set))
        raise ValueError(
            f"LLM judge IDs do not match benchmark; missing={missing}, "
            f"unexpected={unexpected}."
        )
    return [by_id[record_id] for record_id in expected_ids]

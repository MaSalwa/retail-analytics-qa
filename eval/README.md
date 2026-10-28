# Evaluation sets

- `gold_generation.jsonl` — frozen questions, expected numeric values, required text, tolerances, and calculation notes. `expected_display` is the concise answer expected from the query engine, not a prose response that must match exactly. For example, customer and country questions accept `18102` and `United Kingdom` without requiring an unasked-for revenue number.

Notebook `04_evaluate_answers.ipynb` runs the table-Q&A system against this set, scores numeric and factual matches, and can send all answers to Gemini in one judge call. The fixed gold values are based on the cleaned 2009–2010 dataset and the definitions in `src/cleaning.py`.

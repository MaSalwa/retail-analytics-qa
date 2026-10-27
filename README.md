# Retail analytics Q&A with Gemini

Ask natural-language questions of the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) transactions (UK gift-ware sales, 2009–2011) and get grounded numeric answers.

This is a personal learning project. It is built to show two things: **choosing the right architecture for tabular questions**, and **evaluating the system instead of only demoing it**.

## The problem

Typical questions look like:

- How much revenue did the most profitable customer generate?
- What is the average customer lifetime value?
- What was the largest sale?
- Which country generated the most revenue?

Those are **aggregations over a table**, not lookup in a paragraph. A vector RAG over raw invoice lines will retrieve a handful of rows and often invent or miss the global statistic.

## Approach

| Piece | Role |
| --- | --- |
| **Main system** | Clean the Excel file with pandas, then query it with a LlamaIndex pandas/SQL-style engine and **Google Gemini** (free tier). The model writes dataframe operations; pandas computes the number. |
| **Comparison** | A short notebook that runs **naive vector RAG** on the same questions, so the failure mode is visible (wrong chunk, incomplete context, confident but false totals). |
| **Evaluation** | A small gold set with numeric ground truth: exact/tolerance match first, then **Gemini as judge** on the written answer (correctness and faithfulness to the tool output). |

Gemini is the only paid API this repo uses. Embeddings are used only in the naive-RAG comparison, cached locally, and not re-run by default.

## Repo layout (target)

```text
notebooks/
  01_explore_and_clean.ipynb      # schema, missing IDs, cancellations, revenue
  02_table_qa.ipynb               # main system: LlamaIndex + pandas + Gemini
  03_why_naive_rag_fails.ipynb    # same questions through vector RAG
  04_evaluate_answers.ipynb       # gold set, numeric match, LLM-as-judge
src/                              # helpers (LLM client, cleaning, eval)
eval/                             # gold questions and later scores
data/                             # Online Retail II Excel (see data/README.md)
```

## Setup

1. Python 3.11+ recommended.
2. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey).
3. `pip install -r requirements.txt`
4. Work through the notebooks in order.

Do not commit `.env`. Vector stores and embedding caches stay gitignored.

## Dataset

Chen, D. (2015). *Online Retail II* [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

The copy used here is `data/online_retail_II.xlsx`, sheet `Year 2009-2010`.

## What this is not (yet)

- Not a production chatbot or a hosted demo.
- Not a generic “PDF RAG” starter.
- Not using OpenAI.

## Roadmap

1. Explore and clean the sheet.
2. Implement table Q&A with LlamaIndex and Gemini.
3. Show why naive RAG fails on the same gold questions.
4. Evaluate answers (numeric + LLM-as-judge) and write up failure modes.

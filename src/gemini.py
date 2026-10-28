"""Helpers for the retail analytics table-Q&A notebook."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from llama_index.llms.google_genai import GoogleGenAI

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(env_path: str | Path | None = None) -> Path:
    """Load .env from the repo root and return the path used."""
    env_file = Path(env_path) if env_path else REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_file, override=False)
    return env_file


def get_gemini_api_key(env_path: str | Path | None = None) -> str:
    """Return the configured Gemini API key (GOOGLE_API_KEY preferred)."""
    load_project_env(env_path)
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Missing Google Gemini API key. Add GOOGLE_API_KEY to .env."
        )
    return key


def build_gemini_llm(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    env_path: str | Path | None = None,
) -> GoogleGenAI:
    """Create a Gemini LLM configured from project environment variables."""
    api_key = get_gemini_api_key(env_path)
    model_name = model or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
    return GoogleGenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_pandas_query_engine(
    df,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    verbose: bool = False,
    head: int = 20,
    env_path: str | Path | None = None,
) -> PandasQueryEngine:
    """Construct a pandas query engine backed by Gemini."""
    llm = build_gemini_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        env_path=env_path,
    )
    return PandasQueryEngine(df=df, llm=llm, verbose=verbose, head=head)

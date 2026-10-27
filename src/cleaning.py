"""Load and flag Online Retail II transactions. No model calls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = REPO_ROOT / "data" / "online_retail_II.xlsx"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "online_retail_2009_2010.pkl"
SHEET_NAME = "Year 2009-2010"

# Non-product / accounting codes we still keep in the flagged table,
# but exclude from "product sold" style questions by default.
NON_PRODUCT_STOCKCODES = frozenset(
    {
        "POST",
        "DOT",
        "M",
        "m",
        "D",
        "S",
        "C2",
        "B",
        "ADJUST",
        "BANK CHARGES",
        "PADS",
        "AMAZONFEE",
        "TEST001",
    }
)


def load_raw(
    path: Path | str = DEFAULT_XLSX,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Load the 2009–2010 sheet as stored in the Excel file."""
    return pd.read_excel(path, sheet_name=sheet_name)


def add_analysis_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add cancellation, guest, revenue, and stock-type flags, while preserving all rows."""
    out = df.copy()
    out["Invoice"] = out["Invoice"].astype(str)
    out["StockCode"] = out["StockCode"].astype(str)
    out["is_cancellation"] = out["Invoice"].str.startswith("C")
    out["is_guest"] = out["Customer ID"].isna()
    out["is_bad_debt"] = out["StockCode"].eq("B") | out["Invoice"].str.startswith("A")
    out["is_test"] = out["StockCode"].str.upper().str.startswith("TEST")
    out["is_non_product"] = out["StockCode"].isin(NON_PRODUCT_STOCKCODES) | out[
        "StockCode"
    ].str.startswith("gift_")
    out["Revenue"] = out["Quantity"] * out["Price"]
    out["Customer ID"] = out["Customer ID"].astype("Int64")
    return out


def sales_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Positive-quantity, positive-price rows that are not tests or bad-debt adjustments."""
    return df.loc[
        (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["is_test"]
        & ~df["is_bad_debt"]
    ].copy()


def return_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Cancellations or negative quantity (returns / corrections)."""
    return df.loc[df["is_cancellation"] | (df["Quantity"] < 0)].copy()


def identified_customers(df: pd.DataFrame):
    """Customer IDs with at least one positive-price, positive-quantity sale."""
    sold = sales_rows(df)
    return sold.loc[~sold["is_guest"], "Customer ID"].dropna().unique()


def save_processed(df: pd.DataFrame, path: Path | str = PROCESSED_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return path

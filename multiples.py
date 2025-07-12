#!/usr/bin/env python
"""
multiples_to_excel.py
=====================
Build an Excel sheet with the latest valuation ratios for any list of tickers
using *Financial Modeling Prep* (free tier).

Columns
-------
Ticker | Sector | Industry | Date | EV/EBITDA | Current Ratio | P/E | DCF | Price | Price vs DCF %
"""

# ──────────────────────────
# Standard‑library
# ──────────────────────────
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict

# ──────────────────────────
# Third‑party
# ──────────────────────────
import requests
import pandas as pd
import numpy as np

# ──────────────────────────
# Configuration
# ──────────────────────────
API_KEY = "f3i8Pwil0LKmclScLBtTX1fyqLeYu90g"
DELAY_S = 1.0                       # polite gap between API calls

BASE_V3   = "https://financialmodelingprep.com/api/v3"
BASE_STBL = "https://financialmodelingprep.com/stable"

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ──────────────────────────
# Helper: generic GET‑JSON
# ──────────────────────────
def _get_json(url: str, params: Dict[str, str] | None = None) -> list[dict]:
    params = params or {}
    params["apikey"] = API_KEY
    try:
        r = requests.get(url, params=params, timeout=15, headers=HEADERS)
        r.raise_for_status()
        return r.json() or []
    except Exception:
        return []

# ──────────────────────────
# Per‑field fetch helpers
# ──────────────────────────
def _sector_industry(symbol: str) -> tuple[str | float, str | float]:
    data = _get_json(f"{BASE_STBL}/profile", {"symbol": symbol})
    if data:
        return data[0].get("sector", np.nan), data[0].get("industry", np.nan)
    return np.nan, np.nan


def _pe_ttm(symbol: str) -> float:
    data = _get_json(f"{BASE_V3}/key-metrics-ttm/{symbol}")
    return float(data[0].get("peRatioTTM", np.nan)) if data else np.nan


def _ev_cr_date(symbol: str) -> tuple[str | float, float, float]:
    data = _get_json(f"{BASE_STBL}/key-metrics", {"symbol": symbol})
    if data:
        latest = data[0]
        return (
            latest.get("date", np.nan),
            float(latest.get("evToEBITDA", np.nan)),
            float(latest.get("currentRatio", np.nan)),
        )
    return np.nan, np.nan, np.nan


def _dcf_and_price(symbol: str) -> tuple[float, float]:
    """
    Returns (DCF, current stock price) or (NaN, NaN) if unavailable.

    JSON keys can be either 'dcf' & 'Stock Price' (v3) or
    'dcf' & 'stockPrice' (stable).  Handle both.
    """
    data = _get_json(
        f"{BASE_STBL}/discounted-cash-flow",  # stable endpoint
        {"symbol": symbol}
    )
    if data:
        rec = data[0]
        dcf   = rec.get("dcf", np.nan)
        price = rec.get("Stock Price", rec.get("stockPrice", np.nan))
        return float(dcf) if dcf not in (None, "") else np.nan, \
               float(price) if price not in (None, "") else np.nan
    return np.nan, np.nan


def fetch_row(symbol: str) -> Dict[str, object]:
    """Collect all required fields for *one* ticker."""
    sector, industry = _sector_industry(symbol)
    time.sleep(DELAY_S)

    pe = _pe_ttm(symbol)
    time.sleep(DELAY_S)

    date, ev, cr = _ev_cr_date(symbol)
    time.sleep(DELAY_S)

    dcf, price = _dcf_and_price(symbol)
    time.sleep(DELAY_S)

    return {
        "Ticker":        symbol,
        "Sector":        sector,
        "Industry":      industry,
        "Date":          date,
        "EV/EBITDA":     ev,
        "Current Ratio": cr,
        "P/E":           pe,
        "DCF":           dcf,
        "Price":         price,
    }

# ──────────────────────────
# CLI & main
# ──────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch EV/EBITDA, Current Ratio, P/E, DCF & Price "
                    "for tickers via FinancialModelingPrep and save to Excel."
    )
    p.add_argument(
        "-t", "--tickers",
        required=True,
        help="Comma‑separated list of tickers, e.g. AAPL,MSFT,T"
    )
    p.add_argument(
        "-o", "--output",
        default="ratios.xlsx",
        help="Output Excel file (default: ratios.xlsx)"
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    symbols: List[str] = [s.strip().upper() for s in args.tickers.split(",") if s.strip()]

    if not symbols:
        sys.exit("❌  No valid tickers provided.")

    rows = [fetch_row(sym) for sym in symbols]
    df = pd.DataFrame(rows, columns=[
        "Ticker", "Sector", "Industry", "Date",
        "EV/EBITDA", "Current Ratio", "P/E",
        "DCF", "Price"
    ])

    # ───── add %-difference column ─────
    df["Price vs DCF %"] = np.where(
        df["DCF"].notna() & (df["DCF"] != 0),
        (df["Price"] - df["DCF"]) / df["DCF"] * 100,
        np.nan
    ).round(2)

    # re‑order columns
    df = df[
        ["Ticker", "Sector", "Industry", "Date",
         "EV/EBITDA", "Current Ratio", "P/E",
         "DCF", "Price", "Price vs DCF %"]
    ]

    df.to_excel(Path(args.output), index=False)
    print(f"✅  Wrote {len(df)} rows → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

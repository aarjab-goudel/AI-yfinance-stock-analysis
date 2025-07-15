#!/usr/bin/env python
"""
multiples_to_excel.py
---------------------
Create an Excel workbook with the latest valuation / solvency metrics
for a list of tickers using *Financial Modeling Prep* (free tier).

Columns
-------
Ticker | Sector | Industry | Date | EV/EBITDA (TTM) | Current Ratio (TTM) |
P/E (TTM) | ROIC (TTM) | EBITDA/Interest Payments (TTM) | DCF | Price | Price vs DCF %
"""

# ────────────────────────── std‑lib ──────────────────────────
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ───────────────────────── 3rd‑party ─────────────────────────
import requests
import pandas as pd
import numpy as np

# ─────────────────────── Configuration ──────────────────────
API_KEY  = "f3i8Pwil0LKmclScLBtTX1fyqLeYu90g"
DELAY_S  = 1.0
BASE_ST  = "https://financialmodelingprep.com/stable"
HEADERS  = {"User-Agent": "Mozilla/5.0"}

# ─────────────────────── GET‑JSON helper ────────────────────
def _get_json(url: str, params: Dict[str, str] | None = None) -> list[dict]:
    params = params or {}
    params["apikey"] = API_KEY
    try:
        r = requests.get(url, params=params, timeout=15, headers=HEADERS)
        r.raise_for_status()
        return r.json() or []
    except Exception:
        return []

# ─────────────────────── Field helpers ──────────────────────
def _profile(symbol: str) -> Tuple[str, str]:
    data = _get_json(f"{BASE_ST}/profile", {"symbol": symbol})
    if data:
        return data[0].get("sector", np.nan), data[0].get("industry", np.nan)
    return np.nan, np.nan


def _key_metrics_ttm(symbol: str) -> float:
    """ROIC (TTM) – returned as decimal (0.47 → 47 %)."""
    data = _get_json(f"{BASE_ST}/key-metrics-ttm", {"symbol": symbol})
    if data:
        roic = data[0].get("returnOnInvestedCapitalTTM", np.nan)
        return float(roic) if roic not in (None, "") else np.nan
    return np.nan


def _ev_cr_date(symbol: str) -> Tuple[str, float, float]:
    data = _get_json(f"{BASE_ST}/key-metrics", {"symbol": symbol})
    if data:
        rec = data[0]
        return (
            rec.get("date", np.nan),
            float(rec.get("evToEBITDA",  np.nan)),
            float(rec.get("currentRatio", np.nan)),
        )
    return np.nan, np.nan, np.nan


def _ratios_ttm(symbol: str) -> Tuple[float, float]:
    data = _get_json(f"{BASE_ST}/ratios-ttm", {"symbol": symbol})
    if data:
        rec = data[0]
        pe = rec.get("priceToEarningsRatioTTM",   np.nan)
        ic = rec.get("interestCoverageRatioTTM",  np.nan)
        pe = float(pe) if pe not in (None, "") else np.nan
        ic = float(ic) if ic not in (None, "") else np.nan
        return pe, ic
    return np.nan, np.nan


def _dcf_and_price(symbol: str) -> Tuple[float, float]:
    data = _get_json(f"{BASE_ST}/discounted-cash-flow", {"symbol": symbol})
    if data:
        rec   = data[0]
        dcf   = rec.get("dcf", np.nan)
        price = rec.get("Stock Price", rec.get("stockPrice", np.nan))
        dcf   = float(dcf)   if dcf   not in (None, "") else np.nan
        price = float(price) if price not in (None, "") else np.nan
        return dcf, price
    return np.nan, np.nan

# ─────────────────── Consolidated per‑ticker ─────────────────
def fetch_row(symbol: str) -> Dict[str, object]:
    sector, industry = _profile(symbol)
    time.sleep(DELAY_S)

    roic_raw = _key_metrics_ttm(symbol)             # decimal form
    time.sleep(DELAY_S)

    date, ev, curr = _ev_cr_date(symbol)
    time.sleep(DELAY_S)

    pe_ttm, ic_ttm = _ratios_ttm(symbol)
    time.sleep(DELAY_S)

    dcf, price = _dcf_and_price(symbol)
    time.sleep(DELAY_S)

    price_vs_dcf = (price - dcf) / dcf * 100 if dcf not in (0, np.nan) else np.nan

    # ── convert ROIC to pretty percentage string ──
    roic_pct = f"{roic_raw * 100:.2f}%" if roic_raw == roic_raw else np.nan  # check for NaN via self‑equality

    return {
        "Ticker":                         symbol,
        "Sector":                         sector,
        "Industry":                       industry,
        "Date":                           date,
        "EV/EBITDA (TTM)":                ev,
        "Current Ratio (TTM)":            curr,
        "P/E (TTM)":                      pe_ttm,
        "ROIC (TTM)":                     roic_pct,
        "EBITDA/Interest Payments (TTM)": ic_ttm,
        "DCF":                            dcf,
        "Price":                          price,
        "Price vs DCF %":                 round(price_vs_dcf, 2) if price_vs_dcf == price_vs_dcf else np.nan,
    }

# ────────────────────────── CLI & main ───────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch valuation & solvency metrics from FMP and export to Excel."
    )
    p.add_argument("-t", "--tickers", required=True,
                   help="Comma‑separated list of tickers (e.g. AAPL,MSFT).")
    p.add_argument("-o", "--output", default="ratios.xlsx",
                   help="Output Excel file (default: ratios.xlsx).")
    return p.parse_args()


def main() -> None:
    args    = _parse()
    symbols = [s.strip().upper() for s in args.tickers.split(",") if s.strip()]
    if not symbols:
        sys.exit("❌  No valid tickers provided.")

    rows = [fetch_row(sym) for sym in symbols]
    df   = pd.DataFrame(rows, columns=[
        "Ticker", "Sector", "Industry", "Date",
        "EV/EBITDA (TTM)", "Current Ratio (TTM)",
        "P/E (TTM)", "ROIC (TTM)", "EBITDA/Interest Payments (TTM)",
        "DCF", "Price", "Price vs DCF %"
    ])

    df.to_excel(Path(args.output), index=False)
    print(f"✅  Wrote {len(df)} rows → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

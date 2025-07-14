#!/usr/bin/env python
"""
finance_scraper.py
==================
Generate a nicely‑formatted *.txt* financial summary for any Yahoo‑Finance
ticker.

* 4 periods (annual by default, quarterly optional)
* Statements: Balance‑Sheet, Income‑Statement, Cash‑Flow
* Optional (annual‑only) Solvency block:
  Current Ratio | Quick Ratio | Debt/Equity (+ YoY growth)

Usage
-----
$ python finance_scraper.py AAPL
$ python finance_scraper.py AAPL -p quarterly
"""

# ──────────────────────────
# Std‑lib
# ──────────────────────────
import argparse
from pathlib import Path
from typing import Dict, List

# ──────────────────────────
# Third‑party
# ──────────────────────────
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ──────────────────────────
# External API (FinancialModelingPrep) – for solvency ratios
# ──────────────────────────
API_KEY   = "f3i8Pwil0LKmclScLBtTX1fyqLeYu90g"
FMP_BASE  = "https://financialmodelingprep.com/stable"
HEADERS   = {"User-Agent": "Mozilla/5.0"}

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
# yfinance → attribute mapping
# ──────────────────────────
ATTR_MAP = {
    "Balance Sheet":     ("balance_sheet",     "quarterly_balance_sheet"),
    "Income Statement":  ("financials",        "quarterly_financials"),
    "Cash Flow Statement": ("cashflow",        "quarterly_cashflow"),
}

METRIC_SETS: Dict[str, Dict[str, str]] = {
    "Balance Sheet": {
        "Total Assets": "Total Assets",
        "Total Liabilities": "Total Liab",
        "Total Stockholder Equity": "Total Stockholder Equity",
        "Total Debt": "Total Debt",
        "Cash, Cash Equivalents & Short Term Investments": "Cash_STI",
    },
    "Income Statement": {
        "Total Revenue": "Total Revenue",
        "Gross Profit": "Gross Profit",
        "Net Income from Continuing Operation Net Minority Interest": "Net Income",
        "EBITDA": "EBITDA",
        "Research & Development": "Research And Development",
    },
    "Cash Flow Statement": {
        "Free Cash Flow": "Free Cash Flow",
        "Investing Cash Flow": "Investing Cash Flow",
        "Financing Cash Flow": "Financing Cash Flow",
    },
}

# ──────────────────────────
# Helpers
# ──────────────────────────
def _fmt_growth(p: float) -> str:
    """Render growth value or blank for trailing NaN."""
    return "       " if np.isnan(p) else f"{p:+6.2f}%"

def _safe_lookup(df: pd.DataFrame, label: str) -> pd.Series:
    if label in df.index:
        return df.loc[label]
    for idx in df.index:
        if label.lower() in str(idx).lower():
            return df.loc[idx]
    return pd.Series(np.nan, index=df.columns)

# ──────────────────────────
# yfinance extractor
# ──────────────────────────
class YahooExtractor:
    def __init__(self, ticker: str, period: str = "annual") -> None:
        self.ticker = ticker.upper().strip()
        self.period = period
        self._yf = yf.Ticker(self.ticker)

    def _get_stmt(self, attr_pair: tuple, slots: int = 4) -> pd.DataFrame:
        attr = attr_pair[0] if self.period == "annual" else attr_pair[1]
        df: pd.DataFrame = getattr(self._yf, attr, pd.DataFrame())
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.loc[:, df.columns.sort_values(ascending=True)].iloc[:, -slots:]
        df.columns = [c.strftime("%Y-%m-%d") for c in df.columns]
        return df

    def extract(self) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}

        bs = self._get_stmt(ATTR_MAP["Balance Sheet"])
        is_df = self._get_stmt(ATTR_MAP["Income Statement"])
        cf = self._get_stmt(ATTR_MAP["Cash Flow Statement"])

        if not bs.empty:
            bs.loc["Cash_STI"] = _safe_lookup(bs, "Cash").add(
                _safe_lookup(bs, "Short Term Investments"), fill_value=np.nan
            )

        mapping = {"Balance Sheet": bs, "Income Statement": is_df, "Cash Flow Statement": cf}

        for name, src in mapping.items():
            if src.empty:
                cols = bs.columns if not bs.empty else ("",) * 4
                src = pd.DataFrame(index=METRIC_SETS[name].values(), columns=cols)

            req = METRIC_SETS[name]
            rows = pd.DataFrame(index=req.keys(), columns=src.columns)

            for pretty, raw in req.items():
                if raw == "Free Cash Flow":
                    fcf = _safe_lookup(src, "Free Cash Flow")
                    if fcf.notna().any():
                        rows.loc[pretty] = fcf
                    else:
                        ocf = _safe_lookup(src, "Operating Cash Flow")
                        capex = _safe_lookup(src, "Capital Expenditures")
                        rows.loc[pretty] = ocf.sub(capex, fill_value=np.nan)
                else:
                    rows.loc[pretty] = _safe_lookup(src, raw)

            rows = rows.apply(pd.to_numeric, errors="coerce")

            if name == "Balance Sheet" and rows.loc["Total Stockholder Equity"].isna().all():
                rows.loc["Total Stockholder Equity"] = (
                    rows.loc["Total Assets"] - rows.loc["Total Liabilities"]
                )

            nxt = rows.shift(-1, axis=1)
            growth = (nxt - rows).div(rows.abs()).mul(100)
            growth = growth.where(~rows.isna() & (rows != 0))
            growth.index = [f"{idx} Growth" for idx in rows.index]

            out[name] = pd.concat([rows, growth])
        return out

# ──────────────────────────
# Solvency ratios (FMP, annual only)
# ──────────────────────────
def _get_solvency_ratios(tkr: str) -> pd.DataFrame:
    data = _get_json(f"{FMP_BASE}/ratios", {"symbol": tkr, "limit": 4})
    if not data:
        return pd.DataFrame()

    # ensure oldest → newest
    data = sorted(data, key=lambda r: r["date"])
    cols = [rec["date"] for rec in data]

    df = pd.DataFrame(index=["Current Ratio", "Quick Ratio", "Debt to Equity Ratio"],
                      columns=cols)

    df.loc["Current Ratio"]       = [rec.get("currentRatio", np.nan)      for rec in data]
    df.loc["Quick Ratio"]         = [rec.get("quickRatio", np.nan)        for rec in data]
    df.loc["Debt to Equity Ratio"] = [rec.get("debtEquityRatio", np.nan)  for rec in data]

    df = df.apply(pd.to_numeric, errors="coerce")

    nxt = df.shift(-1, axis=1)
    growth = (nxt - df).div(df.abs()).mul(100)
    growth = growth.where(~df.isna() & (df != 0))
    growth.index = [f"{idx} Growth" for idx in df.index]

    return pd.concat([df, growth])

# ──────────────────────────
# Rendering helpers
# ──────────────────────────
def _section(tkr: str, cur: str, title: str, w: int = 117) -> str:
    bar = "=" * w
    return f"{bar} {tkr} (Currency in {cur}) - {title} {bar}\n"

def _render_stmt(name: str, df: pd.DataFrame, tkr: str, cur: str,
                 fmt_int: bool = True) -> str:
    """Generic block renderer – fmt_int=False prints floats with 2 decimals."""
    head = _section(tkr, cur, name.replace("Statement", "Sheet").strip())
    cols = list(df.columns)
    col_fmt = "{:>22}" * len(cols)

    out: List[str] = [head,
                      " " * 45 + col_fmt.format(*cols),
                      "-" * len(head.strip()) + "\n"]

    for idx in df.index:
        if "Growth" not in idx:
            if fmt_int:
                vals = [f"{int(v):,}" if not np.isnan(v) else "NaN" for v in df.loc[idx]]
            else:
                vals = [f"{v:.2f}"      if not np.isnan(v) else "NaN" for v in df.loc[idx]]
            out.append(f"{idx:<42}" + col_fmt.format(*vals))

            g_idx = f"{idx} Growth"
            if g_idx in df.index:
                g_vals = [_fmt_growth(v) for v in df.loc[g_idx]]
                out.append(f"{g_idx:<42}" + col_fmt.format(*g_vals))
            out.append("-" * len(head.strip()))
    out.append(head)
    return "\n".join(out)

def _render_est_link(tkr: str, cur: str) -> str:
    hdr = _section(tkr, cur, "Future Data Analysis for EPS")
    url = f"https://finance.yahoo.com/quote/{tkr}/analysis/"
    return f"{hdr}Future Estimates Data: {url}\n{hdr}"

# ──────────────────────────
# Driver
# ──────────────────────────
def build_txt_report(tkr: str, period: str, out_dir: Path | str = ".") -> Path:
    extractor = YahooExtractor(tkr, period)
    cur = extractor._yf.info.get("financialCurrency", "USD")

    sections = [_render_stmt(name, df, tkr, cur)
                for name, df in extractor.extract().items()]

    # Solvency block – annual only
    if period == "annual":
        solv_df = _get_solvency_ratios(tkr)
        if not solv_df.empty:
            sections.append(_render_stmt("Solvency Ratios", solv_df, tkr, cur,
                                         fmt_int=False))

    sections.append(_render_est_link(tkr, cur))

    suffix = "Q" if period == "quarterly" else "A"
    out_path = Path(out_dir, f"{tkr}_{suffix}_financial_summary.txt")
    out_path.write_text("\n\n".join(sections), encoding="utf-8")
    return out_path

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a 4‑period Yahoo‑Finance summary (annual or quarterly)."
    )
    p.add_argument("ticker", help="Ticker symbol (e.g. AAPL, MSFT)")
    p.add_argument("-p", "--period",
                   choices=("annual", "quarterly"),
                   default="annual",
                   help="Reporting period (default: annual)")
    p.add_argument("-o", "--output", default=".", help="Output directory")
    return p.parse_args()

def main() -> None:
    args = _parse_cli()
    path = build_txt_report(args.ticker.upper(), args.period, args.output)
    print(f"✅  Report written to: {path.resolve()}")

if __name__ == "__main__":
    main()

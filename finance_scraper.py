#!/usr/bin/env python
"""
finance_scraper.py
==================
Generate a nicely formatted *.txt* financial summary for any Yahoo‑Finance
ticker.

* Supported periods: **annual** (default) or **quarterly**.
* Statements covered: Balance‑Sheet, Income‑Statement, Cash‑Flow
  (4 columns each → 3 growth percentages).
* Growth rows show the % change from each column **to the next one** and are
  aligned under the *earlier* period.

Example
-------
$ python finance_scraper.py AAPL                 # 4‑year annual report
$ python finance_scraper.py T --period quarterly # last 4 quarters
"""

# ──────────────────────────
# Std‑lib
# ──────────────────────────
import argparse
from pathlib import Path
from typing import Dict, Any, List

# ──────────────────────────
# 3rd‑party
# ──────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────
# Statement → yfinance attribute mapping
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
# Helper functions
# ──────────────────────────
def _fmt_growth(p: float) -> str:
    """Render growth value or blank for trailing NaN."""
    return "       " if np.isnan(p) else f"{p:+6.2f}%"


def _safe_lookup(df: pd.DataFrame, label: str) -> pd.Series:
    """Exact or fuzzy row lookup; returns NaN series if absent."""
    if label in df.index:
        return df.loc[label]
    for idx in df.index:
        if label.lower() in str(idx).lower():
            return df.loc[idx]
    return pd.Series(np.nan, index=df.columns)


# ──────────────────────────
# Extractor
# ──────────────────────────
class YahooExtractor:
    def __init__(self, ticker: str, period: str = "annual") -> None:
        self.ticker = ticker.upper().strip()
        self.period = period  # 'annual' | 'quarterly'
        self._yf = yf.Ticker(self.ticker)

    # ---------- core pull ----------
    def _get_stmt(self, attr_pair: tuple, slots: int = 4) -> pd.DataFrame:
        """
        Return the last *slots* periods for the requested statement.

        *attr_pair* = (annual_attr, quarterly_attr)
        """
        attr = attr_pair[0] if self.period == "annual" else attr_pair[1]
        df: pd.DataFrame = getattr(self._yf, attr, pd.DataFrame())
        if df is None or df.empty:
            return pd.DataFrame()
        # chronological order: oldest → newest
        df = df.loc[:, df.columns.sort_values(ascending=True)].iloc[:, -slots:]
        df.columns = [c.strftime("%Y-%m-%d") for c in df.columns]
        return df

    # ---------- public ----------
    def extract(self) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}

        # pull statements
        bs = self._get_stmt(ATTR_MAP["Balance Sheet"])
        is_df = self._get_stmt(ATTR_MAP["Income Statement"])
        cf = self._get_stmt(ATTR_MAP["Cash Flow Statement"])

        # synthetic Cash+STI row
        if not bs.empty:
            bs.loc["Cash_STI"] = _safe_lookup(bs, "Cash").add(
                _safe_lookup(bs, "Short Term Investments"), fill_value=np.nan
            )

        mapping = {
            "Balance Sheet": bs,
            "Income Statement": is_df,
            "Cash Flow Statement": cf,
        }

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

            # derive equity if missing
            if name == "Balance Sheet" and rows.loc["Total Stockholder Equity"].isna().all():
                rows.loc["Total Stockholder Equity"] = (
                    rows.loc["Total Assets"] - rows.loc["Total Liabilities"]
                )

            # YoY / QoQ growth – aligned under earlier column
            nxt = rows.shift(-1, axis=1)
            growth = (nxt - rows).div(rows.abs()).mul(100)
            growth = growth.where(~rows.isna() & (rows != 0))
            growth.index = [f"{idx} Growth" for idx in rows.index]

            out[name] = pd.concat([rows, growth])
        return out

# ──────────────────────────
# Formatting
# ──────────────────────────
def _section(tkr: str, cur: str, title: str, w: int = 117) -> str:
    bar = "=" * w
    return f"{bar} {tkr} (Currency in {cur}) - {title} {bar}\n"


def _render_stmt(name: str, df: pd.DataFrame, tkr: str, cur: str) -> str:
    head = _section(tkr, cur, name.replace("Statement", "Sheet").strip())
    cols = list(df.columns)
    col_fmt = "{:>22}" * len(cols)

    out: List[str] = [
        head,
        " " * 45 + col_fmt.format(*cols),
        "-" * len(head.strip()) + "\n",
    ]

    for idx in df.index:
        if "Growth" not in idx:
            vals = [f"{int(v):,}" if not np.isnan(v) else "NaN" for v in df.loc[idx]]
            out.append(f"{idx:<42}" + col_fmt.format(*vals))

            g_idx = f"{idx} Growth"
            if g_idx in df.index:
                g_vals = [_fmt_growth(v) for v in df.loc[g_idx]]
                out.append(f"{g_idx:<42}" + col_fmt.format(*g_vals))
            out.append("-" * len(head.strip()))
    out.append(head)
    return "\n".join(out)

# ────────────────────────
# Add immediately after _render_stmt(...)
# ──────────────────────────
def _render_est_link(tkr: str, cur: str) -> str:
    """Return the header + single‑line URL block."""
    hdr = _section(tkr, cur, "Future Data Analysis for EPS")
    url = f"https://finance.yahoo.com/quote/{tkr}/analysis/"
    return f"{hdr}Future Estimates Data: {url}\n{hdr}"


# ──────────────────────────
# Driver
# ──────────────────────────
def build_txt_report(tkr: str, period: str, out_dir: Path | str = ".") -> Path:
    extractor = YahooExtractor(tkr, period)
    cur = extractor._yf.info.get("financialCurrency", "USD")

    sections = [
        _render_stmt(name, df, tkr, cur) for name, df in extractor.extract().items()
    ]

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
    p.add_argument(
        "-p",
        "--period",
        choices=("annual", "quarterly"),
        default="annual",
        help="Reporting period (default: annual)",
    )
    p.add_argument("-o", "--output", default=".", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    path = build_txt_report(args.ticker, args.period, args.output)
    print(f"✅  Report written to: {path.resolve()}")


if __name__ == "__main__":
    main()

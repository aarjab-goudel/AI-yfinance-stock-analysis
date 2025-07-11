"""
Basic test‑suite for finance_scraper.py

Run with:  pytest
"""
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import finance_scraper as fs


# ─────────────────────────────────────────────
# Fixture: fake yfinance Ticker object
# ─────────────────────────────────────────────
class _DummyYF:
    """Minimal stub replicating the attributes used inside the extractor."""

    def __init__(self):
        idx = pd.to_datetime(["2024-09-30", "2023-09-30", "2022-09-30", "2021-09-30"])
        self.balance_sheet = pd.DataFrame(
            {
                idx[0]: [400, 250, 150, 120, 50],
                idx[1]: [380, 240, 140, 110, 45],
                idx[2]: [350, 230, 120, 105, 40],
                idx[3]: [330, 210, 115, 100, 35],
            },
            index=[
                "Total Assets",
                "Total Liab",
                "Total Stockholder Equity",
                "Total Debt",
                "Cash",
            ],
        )
        # Include short‑term investments so Cash_STI can be calculated
        self.balance_sheet.loc["Short Term Investments"] = [30, 25, 20, 15]

        self.financials = pd.DataFrame(
            {
                idx[0]: [100, 40, 25, 15, 14],
                idx[1]: [90, 35, 22, 13, 11],
                idx[2]: [85, 32, 20, 12, 10],
                idx[3]: [80, 30, 19, 11, 9],
            },
            index=[
                "Total Revenue",
                "Gross Profit",
                "Net Income",
                "EBITDA",
                "Research And Development",
            ],
        )
        self.cashflow = pd.DataFrame(
            {
                idx[0]: [35, -20, -10],
                idx[1]: [32, -18, -9],
                idx[2]: [30, -17, -8],
                idx[3]: [28, -15, -7],
            },
            index=["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"],
        )
        self.info = {"financialCurrency": "USD"}


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────
@mock.patch("yfinance.Ticker")
def test_extract_metrics(mock_yf):
    """Verify that Cash_STI and Free Cash Flow are calculated, and growth rows exist."""
    mock_yf.return_value = _DummyYF()
    extractor = fs.YahooStatementExtractor("DUMMY")

    data = extractor.extract_metrics()
    bs = data["Balance Sheet"]
    cf = data["Cash Flow Statement"]

    # Cash_STI must equal Cash + Short Term Investments in first column
    assert bs.loc["Cash, Cash Equivalents & Short Term Investments"].iloc[0] == 50 + 30

    # Free Cash Flow must equal OCF - CapEx (CapEx absent → NaN)
    assert np.isnan(cf.loc["Free Cash Flow"].iloc[0])

    # Growth rows exist
    assert any("Growth" in row for row in bs.index)


def test_pct_change():
    """Edge‑case handling for percentage change helper."""
    assert fs._pct_change(110, 100) == 10
    assert np.isnan(fs._pct_change(110, 0))
    assert np.isnan(fs._pct_change(np.nan, 100))


def test_format_growth():
    """Proper sign & width."""
    assert fs._format_growth(5.1234).strip() == "+5.12%"
    assert fs._format_growth(-9.9).strip() == "-9.90%"
    assert fs._format_growth(np.nan).strip() == "NaN"

"""Insider trading review fixes: classification of 'public market' trades, duplicates, unknown summary lines stay
unknown, balanced trading is not 'no trades', full refresh keeps insider rows, Ownership shows what it has."""
import pandas as pd
import pytest

from core.screener import insider_activity
from etl import insider_trading as it


@pytest.mark.parametrize("text,kind", [
    ("Sale at price 132.53 per share.", "Sale"),
    ("Disposition in the public market at price 132.53 per share.", "Sale"),
    ("Acquisition in the public market at price 4.42 per share.", "Buy"),
    ("Purchase at price 363.20 per share.", "Buy"),
    ("Stock Gift at price 0.00 per share.", "Gift"),
    ("Conversion of Exercise of derivative security", "Exercise"),
    ("Stock Award(Grant) at price 0.00 per share.", "Award"),
    ("Share buyback program disclosure", "Unknown"),          # 'buy' inside 'buyback' is not a purchase
    ("", "Unknown"),
])
def test_classification(text, kind):
    assert it.classify_transaction(text) == kind


def _fake_ticker(tx=None, summary=None):
    class T:
        insider_transactions = tx
        insider_purchases = summary
    return lambda symbol: T()


def test_identical_rows_are_kept_once(monkeypatch):
    row = {"Text": "Sale at price 10.00 per share.", "Insider": "A", "Position": "CEO", "Shares": 100, "Value": 1000,
           "Start Date": pd.Timestamp("2026-10-01"), "Ownership": "D"}
    monkeypatch.setattr(it.yf, "Ticker", _fake_ticker(tx=pd.DataFrame([row, row, {**row, "Shares": 50}])))
    df = it.extract_insider_transactions(["AAPL"])
    assert len(df) == 2 and sorted(df["shares"]) == [50, 100]


def _summary(lines):
    return pd.DataFrame({"Insider Purchases Last 6m": [l for l, _ in lines], "Shares": [v for _, v in lines]})


def test_a_missing_summary_line_is_unknown_not_zero(monkeypatch):
    monkeypatch.setattr(it.yf, "Ticker", _fake_ticker(summary=_summary([("Purchases", 1000)])))
    s = it.extract_insider_summary(["AAPL"]).iloc[0]
    assert s["insider_purchases_6m"] == 1000 and pd.isna(s["insider_sales_6m"]) and pd.isna(s["net_shares"])
    assert pd.isna(s["pct_buy"]) and pd.isna(s["pct_sell"])
    full = _summary([("Purchases", 1000), ("Sales", 400), ("% Buy Shares", 0.02), ("% Sell Shares", 0.01)])
    monkeypatch.setattr(it.yf, "Ticker", _fake_ticker(summary=full))
    s = it.extract_insider_summary(["AAPL"]).iloc[0]
    assert s["net_shares"] == 600 and s["pct_buy"] == 0.02


def _row(**kw):
    base = dict(insider_purchases_6m=None, insider_sales_6m=None, insider_net_shares_6m=None, insider_pct_buy=None,
                insider_pct_sell=None, insider_extracted_at=pd.Timestamp("2026-10-20"))
    base.update(kw)
    return base


def test_insider_status_never_invents_a_verdict():
    assert insider_activity(_row())["status"] == "NO DATA"
    partial = insider_activity(_row(insider_purchases_6m=1000.0))
    assert partial["status"] == "PARTIAL DATA" and partial["net_pct"] is None and partial["sells"] is None
    flat = insider_activity(_row(insider_purchases_6m=1000.0, insider_sales_6m=1000.0, insider_net_shares_6m=0.0))
    assert flat["status"] == "NET FLAT"
    buy = insider_activity(_row(insider_purchases_6m=900.0, insider_sales_6m=100.0, insider_net_shares_6m=800.0,
                                insider_pct_buy=0.03, insider_pct_sell=0.01))
    assert buy["status"] == "NET BUY" and buy["net_pct"] == pytest.approx(2.0)
    assert buy["as_of"] == pd.Timestamp("2026-10-20")
    one_pct = insider_activity(_row(insider_purchases_6m=900.0, insider_sales_6m=100.0, insider_net_shares_6m=800.0,
                                    insider_pct_buy=0.03))
    assert one_pct["net_pct"] is None                             # one side of the % is unknown: no net %


def test_full_refresh_carries_insider_tables_forward():
    from etl.pipeline import CARRY_FORWARD_TABLES
    assert {"raw.insider_summary", "raw.insider_transactions"} <= set(CARRY_FORWARD_TABLES)


class _UI:
    def __init__(self):
        self.text, self.frames = [], []

    def __getattr__(self, name):
        def call(*a, **k):
            if a:
                self.text.append(str(a[0]))
            return self
        return call

    def columns(self, n):
        return [self] * n

    def expander(self, *a, **k):
        self.text.append(str(a[0]) if a else "")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def dataframe(self, df, **k):
        self.frames.append(df)


def _ownership(monkeypatch, insider, tx):
    from types import SimpleNamespace as NS
    from views.stock_analysis import ownership
    import services.db as db
    ui = _UI()
    monkeypatch.setattr(ownership, "st", ui)
    monkeypatch.setattr(db, "load_insider_transactions", lambda **k: tx)
    ownership._render_insider_activity(NS(ticker="AAPL", scores={"insider": insider}))
    return ui


def test_ownership_shows_stored_transactions_without_a_summary_and_names_usd(monkeypatch):
    tx = pd.DataFrame({"transaction_date": [pd.Timestamp("2026-10-01")], "insider_name": ["A"], "position": ["CEO"],
                       "transaction_type": ["Buy"], "shares": [100], "value": [12000.0]})
    ui = _ownership(monkeypatch, {"status": "NO DATA"}, tx)
    assert any("No insider-trading summary" in t for t in ui.text)
    assert ui.frames and "value (USD)" in ui.frames[0].columns


def test_ownership_says_a_failed_read_and_an_empty_one_apart(monkeypatch):
    bad = pd.DataFrame()
    bad.attrs["error"] = "IOException: disk"
    text = " ".join(_ownership(monkeypatch, {"status": "NET BUY", "buys": 1.0, "sells": 0.0, "net_pct": None},
                               bad).text)
    assert "could not be read" in text
    text = " ".join(_ownership(monkeypatch, {"status": "PARTIAL DATA", "buys": 5.0, "sells": None}, pd.DataFrame()).text)
    assert "No insider transactions stored" in text and "PARTIAL DATA" in text

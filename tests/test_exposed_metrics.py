"""Metrics that were computed but not shown (2026-10-16): measured Quality inputs, insider activity, DCF model risk,
analyst detail, per-ticker data confidence, history jumps, per-holding sensitivities and portfolio factor tilts."""
import os

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    from tests.synthetic_warehouse import build
    return build(str(tmp_path_factory.mktemp("wh") / "stock_dw.duckdb"))


@pytest.fixture(scope="module")
def table(warehouse):
    from core.screener import build_screener_table
    from services.db import read_warehouse
    with duckdb.connect(warehouse, read_only=True) as c:
        f = read_warehouse(c)
    return build_screener_table(f[1], f[0], f[4], f[3], f[7], {}, f[13] if len(f) > 13 else None)


def test_screener_carries_the_measured_quality_inputs(table):
    for col in ("ROIC (%)", "Net Debt/EBITDA", "FCF Conversion (x)", "Insider 6M (%)", "_metrics", "_insider"):
        assert col in table.columns
    m = table.set_index("Ticker").at["AAPL", "_metrics"]
    assert {"roic_pct", "fcf_conversion", "net_debt_ebitda", "rev_cagr_pct", "profit_years_pct", "margin_vol_pp"} <= set(m)
    assert m["rev_cagr_pct"] == pytest.approx(3.0, abs=0.01)          # synthetic revenue grows 3% a year
    assert m["profit_years_pct"] == 100.0


def test_scanner_offers_the_new_columns():
    from views.scanner import EXTRA_COLS
    assert {"ROIC (%)", "Net Debt/EBITDA", "FCF Conversion (x)", "Insider 6M (%)"} <= set(EXTRA_COLS)


def test_insider_activity_tells_no_data_from_no_trades():
    from core.screener import insider_activity
    assert insider_activity({"insider_purchases_6m": 0, "insider_sales_6m": 0, "insider_net_shares_6m": 0,
                             "insider_pct_buy": 0, "insider_pct_sell": 0})["status"] == "NO DATA"
    buy = insider_activity({"insider_purchases_6m": 108247, "insider_sales_6m": 53597, "insider_net_shares_6m": 54650,
                            "insider_pct_buy": 0.046, "insider_pct_sell": 0.023})
    assert buy["status"] == "NET BUY" and buy["net_pct"] == pytest.approx(2.3)
    assert insider_activity({"insider_purchases_6m": 0, "insider_sales_6m": 500, "insider_net_shares_6m": -500,
                             "insider_pct_buy": 0, "insider_pct_sell": 0.01})["status"] == "NET SELL"


def test_confidence_items_flag_what_the_reader_should_doubt():
    from ui.provenance import confidence_items, pit_caption, stale_fields
    prov = {"stale": pd.DataFrame({"stale_fields": ["revenue_ttm@2026-10-07,total_debt@2026-10-07"]}),
            "payout": pd.DataFrame({"payout_status": ["PARTIAL"], "payout_note": ["issuance PARTIAL"]}),
            "sources": pd.DataFrame({"check_name": ["price_last_close"], "period": ["2026-10-15"], "ours": [100.0],
                                     "theirs": [103.0], "diff_pct": [-2.9], "status": ["MISMATCH"],
                                     "source": ["TradingView"], "detail": ["x"]}),
            "exceptions": pd.DataFrame({"stage": ["history_audit"], "reason": ["unexplained price jump(s)"],
                                        "detail": ["1 day(s), worst -78.0% on 2024-12-09"]}),
            "annual_pit": pd.DataFrame({"date": ["2025-12-31"], "available_date": ["2026-02-12"],
                                        "available_source": ["SEC_FILING"]})}
    assert stale_fields(prov) == [("revenue (TTM)", "2026-10-07"), ("debt", "2026-10-07")]
    text = " | ".join(t for _, t in confidence_items(prov, {"model": "dcf", "terminal_share": 0.82}))
    for part in ("earlier extraction", "payout evidence PARTIAL", "TradingView disagrees", "price jump", "82%"):
        assert part in text
    assert "2026-02-12 (SEC filing)" in pit_caption(prov, "annual")
    clean = {"sources": prov["sources"].assign(status="MATCH")}
    assert confidence_items(clean, {"model": "dcf", "terminal_share": 0.5}) == [("info", "matches TradingView")]


def test_portfolio_tilts_show_a_high_beta_book(warehouse):
    from core.factor_test import portfolio_tilts
    from services.db import read_warehouse
    with duckdb.connect(warehouse, read_only=True) as c:
        f = read_warehouse(c)
    comp = f[1].copy()
    comp["beta"] = comp["ticker"].map({"NVDA": 2.5, "AAPL": 1.2, "MSFT": 1.0, "JNJ": 0.5, "KO": 0.6, "SAP.DE": 1.0})
    t = portfolio_tilts({"NVDA": 0.5, "KO": 0.1}, comp, f[0]).set_index("factor")
    assert set(t.index) == {"beta", "size", "value", "momentum"}
    assert t.at["size", "tilt_sd"] is not None and t.at["size", "coverage"] == pytest.approx(1.0)


@pytest.mark.smoke
def test_stock_analysis_shows_measures_confidence_and_model_risk(warehouse, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import services.db as db
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv("STOCK_DW_PATH", warehouse)
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setattr(db, "DB_PATH", warehouse)
    db.load_ticker_provenance.clear()
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=300)
    at.secrets["SUPABASE_URL"], at.secrets["SUPABASE_KEY"] = "http://127.0.0.1:9", "test-key"
    at.secrets["COOKIE_SECRET"] = "test-secret"
    at.session_state["authenticated"] = True
    at.session_state["user_id"] = "00000000-0000-0000-0000-000000000000"
    at.session_state["active_tab"] = "🔬 Stock Analysis"
    at.session_state["deep_ticker_selector"] = "NVDA"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "ROIC (3y median)" in html and "FCF conversion" in html and "Net debt / EBITDA" in html
    assert "Data confidence" in html and "TradingView disagrees" in html     # NVDA's stubbed quote is 5% off
    assert "Insider activity" in html
    if "Intrinsic Valuation (FCFE DCF)" in html and "No positive free cash flow" not in " ".join(i.value for i in at.info):
        assert "Value after year" in html and "Net debt repayment / FCFE" in html
    captions = " ".join(c.value for c in at.caption)
    assert "This quarter" in captions or "No analyst estimates" in " ".join(i.value for i in at.info)

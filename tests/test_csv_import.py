"""Portfolio CSV/Excel staging: fake session and FX, no network or warehouse writes."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import date
import io

import duckdb
import pandas as pd
import pytest

from services import market_data
from views import portfolio


RATE_DATE = date(2026, 10, 22)


@pytest.fixture
def session(monkeypatch):
    class State(dict):
        def __getattr__(self, name):            # AttributeError (not KeyError) so copy.deepcopy can probe it
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name) from None
        __setattr__ = dict.__setitem__

    state = State(portfolio_tickers=["OLD"], portfolio_shares={"OLD": 2.0},
                  portfolio_cost={"OLD": 50.0}, portfolio_cash_eur=30.0,
                  portfolio_stops={"OLD": 40.0}, portfolio_targets={}, _portfolio_version=3)
    monkeypatch.setattr(portfolio.st, "session_state", state)
    monkeypatch.setattr(portfolio, "stored_fx_rate",
                        lambda *a, **kw: pytest.fail("unexpected FX lookup"))
    return state


@pytest.fixture
def fx(session, monkeypatch):
    calls = []
    rates = {"USD": (0.89, RATE_DATE), "GBP": (1.17, RATE_DATE),
             "ZAR": (0.05, RATE_DATE), "ILS": (0.24, RATE_DATE)}

    def lookup(ccy, *, with_date=False):
        assert with_date is True
        calls.append(ccy)
        return rates.get(ccy)

    monkeypatch.setattr(portfolio, "stored_fx_rate", lookup)
    return rates, calls


@pytest.mark.parametrize("column", [None, "currency", "CCY", "Cost_Currency"])
@pytest.mark.parametrize("value", [None, "", "  ", "eur"])
def test_legacy_and_blank_currency_remain_eur(session, column, value):
    row = {"symbol": "aapl", "shares": 2, "cost_basis": 100}
    if column:
        row[column] = value
    details = {}
    valid, missing = portfolio._stage_import(pd.DataFrame([row]), ["AAPL"], "Overwrite", details)
    assert (valid, missing) == (["AAPL"], [])
    assert session.portfolio_shares == {"AAPL": 2}
    assert session.portfolio_cost == {"AAPL": 100}
    assert details == {"ambiguous": [], "conversions": {}}


@pytest.mark.parametrize("currency,expected,major", [
    ("USD", 89, "USD"), (" usd ", 89, "USD"), ("GBP", 117, "GBP"),
    ("GBp", 1.17, "GBP"), ("GBX", 1.17, "GBP"), ("GBx", 1.17, "GBP"),
    ("ZAc", 0.05, "ZAR"), ("ILA", 0.24, "ILS"),
])
@pytest.mark.parametrize("column", ["Currency", "ccy", "COST_CURRENCY"])
def test_cost_conversion_and_dated_details(session, fx, currency, expected, major, column):
    details = {}
    portfolio._stage_import(pd.DataFrame([{"ticker": "AAPL", "shares": 2, "cost": 100,
                                          column: currency}]), ["AAPL"], "Overwrite", details)
    assert session.portfolio_shares == {"AAPL": 2}
    assert session.portfolio_cost["AAPL"] == pytest.approx(expected)
    assert details["conversions"][currency.strip()] == (pytest.approx(expected / 100), RATE_DATE)
    assert fx[1] == [major]


@pytest.mark.parametrize("mode", ["Overwrite", "Merge (Accumulate)"])
@pytest.mark.parametrize("currency,quote", [
    ("XYZ", None), ("JPY", None), ("USD", (0.0, RATE_DATE)),
    ("USD", (-1.0, RATE_DATE)), ("USD", (float("nan"), RATE_DATE)),
    ("USD", (float("inf"), RATE_DATE)), ("USD", (0.89, None)),
])
def test_fx_error_in_later_row_leaves_entire_state_unchanged(session, fx, mode, currency, quote):
    fx[0][currency] = quote
    before = deepcopy(session)
    rows = [{"ticker": "AAPL", "shares": 1, "cost": 10},
            {"ticker": "MSFT", "shares": 2, "cost": 100, "currency": currency}]
    with pytest.raises(ValueError, match=f"{currency}.*MSFT"):
        portfolio._stage_import(pd.DataFrame(rows), ["AAPL", "MSFT"], mode)
    assert session == before


@pytest.mark.parametrize("tickers,raw,expected,ambiguous", [
    (["SAP.DE", "SAP.F"], "SAP", [], ["SAP → SAP.DE, SAP.F"]),
    (["SAP.DE"], "SAP", ["SAP.DE"], []),
    (["SAP.DE", "SAP.F", "SAP"], "sap", ["SAP"], []),
    (["NVO", "NOVO B.DE", "NOVO B.F"], "NOVO B", ["NVO"], []),
])
def test_ticker_resolution(session, tickers, raw, expected, ambiguous):
    before = deepcopy(session)
    details = {}
    valid, missing = portfolio._stage_import(
        pd.DataFrame([{"ticker": raw, "shares": 1, "cost": 10}]), tickers, "Overwrite", details)
    assert (valid, missing) == (expected, [])
    assert details["ambiguous"] == ambiguous
    if not expected:
        assert session == before


def test_ambiguous_rows_skipped_while_supported_rows_are_staged(session):
    details = {}
    rows = [{"ticker": t, "shares": 1, "cost": 10} for t in ["SAP", "AAPL", "ZZZ"]]
    valid, missing = portfolio._stage_import(pd.DataFrame(rows), ["SAP.F", "SAP.DE", "AAPL"],
                                            "Overwrite", details)
    assert (valid, missing) == (["AAPL"], ["ZZZ"])
    assert session.portfolio_tickers == ["AAPL"]
    assert details["ambiguous"] == ["SAP → SAP.DE, SAP.F"]


@pytest.mark.parametrize("mode", ["Overwrite", "Merge (Accumulate)"])
def test_duplicate_lots_convert_before_averaging_and_discard_restores_book(session, fx, mode):
    before = deepcopy(session)
    rows = [{"ticker": "AAPL", "shares": 1, "cost": 100},
            {"ticker": "AAPL", "shares": 3, "cost": 100, "currency": "USD"}]
    portfolio._stage_import(pd.DataFrame(rows), ["AAPL"], mode)
    assert session.portfolio_shares["AAPL"] == 4
    assert session.portfolio_cost["AAPL"] == pytest.approx((100 + 3 * 89) / 4)
    assert ("OLD" in session.portfolio_tickers) == mode.startswith("Merge")
    # A second draft import must retain the first saved book.
    portfolio._stage_import(pd.DataFrame([rows[0]]), ["AAPL"], "Merge (Accumulate)")
    assert session._import_saved_book == {k: before[k] for k in portfolio._IMPORT_BOOK}
    portfolio._discard_import()
    assert {k: session[k] for k in portfolio._IMPORT_BOOK} == {
        k: before[k] for k in portfolio._IMPORT_BOOK}


def test_merge_converts_cost_before_averaging_with_existing_holding(session, fx):
    portfolio._stage_import(pd.DataFrame([{"ticker": "OLD", "shares": 2, "cost": 100,
                                          "currency": "USD"}]), ["OLD"], "Merge (Accumulate)")
    assert session.portfolio_shares == {"OLD": 4}
    assert session.portfolio_cost["OLD"] == pytest.approx((2 * 50 + 2 * 89) / 4)


@pytest.mark.parametrize("csv,qty,cost", [
    ("ticker\nAAPL\n", None, None),
    ("ticker,shares,cost\nAAPL,0,0\n", 0, 0),
    ("ticker,shares,cost\nAAPL,2,\n", 2, None),
    ("ticker,shares,cost\nAAPL,,100\n", None, 100),
])
def test_legacy_csv_preserves_unknown_and_zero(session, csv, qty, cost):
    portfolio._stage_import(pd.read_csv(io.StringIO(csv)), ["AAPL"], "Overwrite")
    actual_qty, actual_cost = session.portfolio_shares["AAPL"], session.portfolio_cost["AAPL"]
    assert pd.isna(actual_qty) if qty is None else actual_qty == qty
    assert pd.isna(actual_cost) if cost is None else actual_cost == cost


@pytest.mark.parametrize("field", ["shares", "cost"])
def test_negative_value_in_later_row_is_still_atomic(session, field):
    before = deepcopy(session)
    rows = [{"ticker": "AAPL", "shares": 1, "cost": 100},
            {"ticker": "MSFT", "shares": 1, "cost": 100, field: -1}]
    with pytest.raises(ValueError, match="Invalid shares or cost for MSFT"):
        portfolio._stage_import(pd.DataFrame(rows), ["AAPL", "MSFT"], "Overwrite")
    assert session == before


def test_stored_fx_preserves_scalar_api_and_returns_latest_positive_rate_date(monkeypatch):
    # In-memory fake warehouse only; all connections to the project warehouse are replaced.
    with duckdb.connect() as conn:
        conn.execute("CREATE SCHEMA raw")
        conn.execute("CREATE TABLE raw.stock_prices (date DATE, currency VARCHAR, fx_rate DOUBLE)")
        conn.executemany("INSERT INTO raw.stock_prices VALUES (?, ?, ?)", [
            ("2026-10-20", "USD", 0.80), ("2026-10-22", "usd", 0.89),
            ("2026-10-23", "USD", 0.0), ("2026-10-23", "USD", -1.0),
            ("2026-10-22", "GBP", 1.17), ("2026-10-22", "JPY", None)])

        def connection(*, read_only):
            assert read_only is True
            return nullcontext(conn)

        monkeypatch.setattr(market_data, "get_db_connection", connection)
        monkeypatch.setattr(market_data, "_stored_fx", market_data._stored_fx.__wrapped__)
        assert market_data.stored_fx_rate("USD") == pytest.approx(0.89)
        assert market_data.stored_fx_rate("USD", with_date=True) == (0.89, RATE_DATE)
        assert market_data.stored_fx_rate("GBp") == pytest.approx(0.0117)
        assert market_data.stored_fx_rate("GBX", with_date=True) == (pytest.approx(0.0117), RATE_DATE)
        assert market_data.stored_fx_rate("EUR") == 1.0
        assert market_data.stored_fx_rate("JPY", with_date=True) is None
        assert market_data.stored_fx_rate("XYZ", with_date=True) is None

    # A7-04: stored ZAc, padded major codes, and unknown FX preserve the scalar API.
    from services import market_data as md
    monkeypatch.setattr(md, "_stored_fx", lambda ccy, generation=None: (0.05, "2026-10-23") if ccy == "ZAR" else None)
    fn = getattr(md.stored_fx_rate, "__wrapped__", md.stored_fx_rate)
    assert fn("ZAc") == pytest.approx(0.0005) and fn(" ZAR ") == pytest.approx(0.05) and fn("XXX") is None


# A7-04: stored minor-unit aliases resolve to their major currency.
def test_stored_fx_finds_a_rate_stored_under_the_minor_unit(monkeypatch):
    import contextlib
    from services import market_data as md
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.stock_prices (date DATE, currency VARCHAR, fx_rate DOUBLE)")
    con.execute("INSERT INTO raw.stock_prices VALUES ('2026-10-23', ' ZAc ', 0.05), ('2026-10-23', 'ILA', 0.25), "
                "('2026-10-22', 'GBX', 1.17)")
    monkeypatch.setattr(md, "get_db_connection", lambda read_only=True: contextlib.nullcontext(con))
    assert md._stored_fx("ZAR")[0] == pytest.approx(0.05)
    assert md._stored_fx("ILS")[0] == pytest.approx(0.25)
    assert md._stored_fx("GBP")[0] == pytest.approx(1.17)
    assert md._stored_fx("USD") is None

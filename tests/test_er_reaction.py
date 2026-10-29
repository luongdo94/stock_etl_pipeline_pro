import pandas as pd

from core.er_reaction import live_reaction, window_headlines, check_drivers, direction_line


def test_live_after_hours_and_pre_market():
    event = '2026-10-28T20:30:00Z'
    q = {'regularMarketPrice': 100, 'regularMarketTime': int(pd.Timestamp('2026-10-28T20:00:00Z').timestamp()),
         'postMarketPrice': 105, 'postMarketTime': int(pd.Timestamp('2026-10-28T21:00:00Z').timestamp())}
    result = live_reaction(q, event, 'AAPL')
    assert result['label'] == 'after hours'
    assert round(result['return'], 2) == .05
    q.update(postMarketPrice=None, preMarketPrice=103, preMarketTime=int(pd.Timestamp('2026-10-29T12:00:00Z').timestamp()))
    assert live_reaction(q, event, 'AAPL')['label'] == 'pre-market'


def test_live_requires_post_release_quote_and_fields():
    q = {'regularMarketPrice': 100, 'regularMarketTime': 1793217600,
         'postMarketPrice': 105, 'postMarketTime': int(pd.Timestamp('2026-10-28T20:20:00Z').timestamp())}
    assert live_reaction(q, '2026-10-28T20:30:00Z', 'AAPL')['return'] is None
    assert live_reaction({}, '2026-10-28T20:30:00Z', 'AAPL')['return'] is None


def test_headline_window_and_driver_checks():
    headlines = window_headlines([{'title': 'Orders up 12%', 'published': '2026-10-29T10:00:00Z', 'source': 'News', 'link': 'https://example.com'},
                                  {'title': 'Old', 'published': '2026-10-28T19:00:00Z'},
                                  {'title': 'Late', 'published': '2026-11-02T00:00:00Z'}], '2026-10-28T20:30:00Z', 'AAPL')
    assert len(headlines) == 1 and headlines[0]['id'] == 'H1'
    lines = [{'label': 'Revenue', 'shown': '$10B', 'value': 10}]
    drivers, dropped = check_drivers([{'text': 'Revenue was $10B', 'evidence': ['Revenue']},
        {'text': 'Orders rose 12%', 'evidence': ['H1']}, {'text': 'No source', 'evidence': []},
        {'text': 'Revenue was 99', 'evidence': ['Revenue']}, {'text': 'Buy shares', 'evidence': ['Revenue']},
        {'text': 'Late', 'evidence': ['H2']}], lines, headlines, .05)
    assert len(drivers) == 2
    assert len(dropped) == 4


def test_direction_is_computed():
    line = direction_line([{'metric': 'revenue', 'marker': 'beat', 'diff_pct': .1}, {'metric': 'eps', 'diff_pct': .1}], -.05, 'English')
    assert 'beat' in line and '-5.0%' in line


def test_before_open_session_uses_previous_close():
    q = {'regularMarketPrice': 95, 'regularMarketPreviousClose': 100,
         'regularMarketTime': int(pd.Timestamp('2026-10-29T15:00:00Z').timestamp())}
    result = live_reaction(q, '2026-10-29T12:00:00Z', 'AAPL')
    assert result['label'] == "today's session" and round(result['return'], 2) == -.05


def test_live_fetch_collects_quote(monkeypatch):
    import yfinance as yf
    from services import earnings as svc
    class FakeTicker:
        def __init__(self, ticker):
            pass
        @property
        def info(self):
            return {'regularMarketPrice': 100, 'postMarketPrice': 105, 'currency': 'USD'}
    monkeypatch.setattr(yf, 'Ticker', FakeTicker)
    monkeypatch.setattr(svc, '_forget_yahoo_cache', lambda: None)
    got = svc.fetch_live_report('AAPL', '', earnings_dates=lambda _: None, release=lambda *_: (None, 'none', None))
    assert got['quote']['postMarketPrice'] == 105 and got['quote']['currency'] == 'USD'


def test_stale_or_post_open_extended_quote_is_not_used():
    close = int(pd.Timestamp('2026-10-28T20:00:00Z').timestamp())
    q = {'regularMarketPrice': 100, 'regularMarketTime': close, 'preMarketPrice': 103,
         'preMarketTime': int(pd.Timestamp('2026-10-29T15:00:00Z').timestamp())}
    assert live_reaction(q, '2026-10-28T20:30:00Z', 'AAPL')['return'] is None
    q['preMarketTime'] = int(pd.Timestamp('2026-10-30T12:00:00Z').timestamp())
    q['regularMarketTime'] = int(pd.Timestamp('2026-10-29T20:00:00Z').timestamp())
    assert live_reaction(q, '2026-10-28T20:30:00Z', 'AAPL')['return'] is None



def test_the_first_session_after_an_after_close_release_is_measured():
    """FTNT reported at 16:00 ET on 28 Oct 2026 (stamped at the close) and traded at $137.00 the next day against a
    $174.08 close: -21.3%, not N/A. A quote two weekdays later is refused (its previous close is not the release day's)."""
    from core import er_reaction as r
    release = pd.Timestamp("2026-10-28 20:00", tz="UTC")
    def quote(ts):
        return {"regularMarketPrice": 137.0, "regularMarketTime": int(pd.Timestamp(ts, tz="America/New_York").timestamp()),
                "regularMarketPreviousClose": 174.08, "currency": "USD"}
    got = r.live_reaction(quote("2026-10-29 11:00"), release, "FTNT")
    assert round(got["return"], 4) == round(137.0 / 174.08 - 1, 4) and got["label"] == "next session (intraday)"
    assert r.live_reaction(quote("2026-10-29 16:05"), release, "FTNT")["label"] == "next session close"
    assert r.live_reaction(quote("2026-10-30 11:00"), release, "FTNT")["return"] is None



def test_a_headline_id_in_the_text_is_not_read_as_a_figure():
    """Real FTNT reply: "...while H9 reports strong results" was dropped because H9 read as the number 9."""
    lines = [{"label": "Total revenue", "shown": "$1.70B", "change": "+20% YoY"}]
    heads = [{"id": "H9", "title": "Fortinet posts strong results and raises guidance", "source": "x",
              "published": "2026-10-29T12:00:00+00:00", "link": "https://example.com"}]
    kept, dropped = check_drivers([{"text": "Revenue grew 20% YoY and H9 reports strong results, yet the stock moved -21.6%.",
                                    "evidence": ["Total revenue", "H9"]}], lines, heads, -0.216)
    assert len(kept) == 1 and not dropped
    kept, dropped = check_drivers([{"text": "Revenue grew 25% YoY.", "evidence": ["Total revenue"]}], lines, heads, -0.216)
    assert not kept and dropped[0]["reason"] == "a figure not in the cited evidence"

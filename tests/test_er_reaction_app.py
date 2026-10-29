"""The single-stock live-to-sheet path uses only offline providers."""
import json

import pandas as pd

from tests.test_app_smoke import _app, warehouse  # noqa: F401,F811


def test_live_fetch_then_sheet_shows_reaction_and_headline(warehouse, monkeypatch):
    from services import earnings as svc, news
    from views import earnings as view
    from core.release_figures import headline_figures

    event = pd.Timestamp('2026-10-28T20:30:00Z')
    events = pd.DataFrame([{'ticker': 'AAPL', 'earnings_ts': event, 'eps_actual': 2.0,
                            'eps_estimate': 1.8, 'surprise_pct': .11, 'direction': 1,
                            'small_base': False, '_extracted_at': pd.Timestamp('2026-10-28T21:00:00Z')}])
    text = 'Revenue was $10 billion. Earnings release for the quarter ended September 30, 2026.'
    quote = {'regularMarketPrice': 100, 'regularMarketTime': int(pd.Timestamp('2026-10-28T20:00:00Z').timestamp()),
             'postMarketPrice': 95, 'postMarketTime': int(pd.Timestamp('2026-10-28T21:00:00Z').timestamp()),
             'currency': 'USD'}
    monkeypatch.setattr(svc, 'load_earnings_events', lambda *_: events.copy())
    monkeypatch.setattr(svc, 'fetch_live_report', lambda *a, **k: {
        'ticker': 'AAPL', 'fetched_at': pd.Timestamp('2026-10-28T21:10:00Z'), 'events': events.copy(),
        'events_error': None, 'release': (text, 'https://example.com/release', '2026-10-28'),
        'figures': headline_figures(text), 'quote': quote, 'quote_error': None})
    monkeypatch.setattr(news, 'fetch_news_sentiment', lambda *a, **k: {'error': None, 'history_error': None, 'yahoo': {}})
    monkeypatch.setattr(news, 'stored_headlines', lambda *a, **k: [{'title': 'Orders increased', 'source': 'Example',
        'published': '2026-10-28T21:05:00Z', 'link': 'https://example.com/news'}])
    monkeypatch.setattr(view, '_previous_guidance', lambda *a: ([], None))
    def model(prompt, *args, **kwargs):
        if kwargs.get('schema'):
            return json.dumps({'drivers': [{'text': 'Orders increased', 'evidence': ['H1']}]})
        return json.dumps({'sections': [], 'key_updates': [], 'guidance': [], 'points': []})
    monkeypatch.setattr(svc, 'generate_er_sheet', model)
    at = _app('🗓️ Earnings')
    at.session_state['active_ticker'] = 'AAPL'
    at.run()
    at.selectbox(key='er_section').select('Single stock').run()
    at.button(key='er_live_btn_AAPL').click().run()
    at.button(key='er_sheet_build').click().run()
    assert not at.exception, [e.value for e in at.exception]
    html = ' '.join(m.value for m in at.markdown)
    assert 'Price reaction and possible drivers' in html and '-5.0%' in html
    assert 'Orders increased' in html and 'https://example.com/news' in html

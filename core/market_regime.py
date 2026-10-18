"""Market-wide context for the dashboard header — pure functions, no Streamlit.

Horizon window, movers, hot alerts, breadth and the 0-100 market-confidence score / regime.
The regime is a heuristic (not validated against returns); the UI says so.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.rating import QUALITY_TIERS

# non-investable series, kept out of the stock universe: ^TNX is the 10Y yield (rate sensitivities); DX-Y.NYB, ^IRX, CL=F
# and GC=F are market-context inputs (dollar, 3-month bill, oil, gold)
INDICES = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC", "^TNX", "DX-Y.NYB", "^IRX", "CL=F", "GC=F"]
BREADTH_EXCLUDE = INDICES + ["^TNX", "^IRX"]

ELITE, WEAK = QUALITY_TIERS[0][0], QUALITY_TIERS[-1][0]    # Quality tier cut-offs (core/rating.py)
HORIZON_DAYS = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095, "5Y": 1825}


def horizon_start(horizon, min_date, max_date):
    """Start date of a sidebar horizon ("Custom" is resolved by the caller)."""
    if horizon in HORIZON_DAYS:
        return max(max_date - timedelta(days=HORIZON_DAYS[horizon]), min_date)
    if horizon == "YTD":
        return max(date(max_date.year, 1, 1), min_date)
    if horizon == "ALL":
        return min_date
    return max(max_date - timedelta(days=365), min_date)


STALE_DAYS = 4          # a ticker whose last close is this many calendar days behind the newest close is stale


def movers(prices_full, top=5):
    """Change between each ticker's OWN last two sessions → (movers_df, gainers, losers). movers_df: ticker, price_close,
    price_close_prev, chg_24h (%), last_date, prev_date, stale. A market closed on the newest date (holiday, other time
    zone) still has its own last move; a ticker more than STALE_DAYS behind is stale and never a mover (review SIG-06:
    the two newest dates of the whole universe dropped such tickers, and a missing move later read as 0)."""
    stocks = prices_full[~prices_full["ticker"].isin(INDICES)][["date", "ticker", "price_close"]]
    if stocks.empty:
        empty = pd.DataFrame(columns=["ticker", "price_close", "price_close_prev", "chg_24h", "last_date", "prev_date",
                                      "stale"])
        return empty, empty, empty
    last2 = stocks.sort_values(["ticker", "date"]).groupby("ticker").tail(2)
    cur = last2.groupby("ticker").tail(1).set_index("ticker")
    prev = last2.groupby("ticker").head(1).set_index("ticker")
    m = pd.DataFrame({"price_close": cur["price_close"], "last_date": cur["date"],
                      "price_close_prev": prev["price_close"], "prev_date": prev["date"]}).reset_index()
    two = m["prev_date"] < m["last_date"]                       # a ticker with one session has no move
    ok = two & (m["price_close_prev"] > 0)
    m["chg_24h"] = np.where(ok, (m["price_close"] / m["price_close_prev"] - 1) * 100, np.nan)
    newest = pd.to_datetime(m["last_date"]).max()
    m["stale"] = (newest - pd.to_datetime(m["last_date"])).dt.days > STALE_DAYS
    live = m[~m["stale"] & m["chg_24h"].notna()]
    return (m, live.sort_values("chg_24h", ascending=False).head(top),
            live.sort_values("chg_24h", ascending=True).head(top))


# Observation labels (review SIG-04: "EXIT / RISK" from an RSI alone read as a sell instruction). They describe the
# tape; the Decision is the only recommendation, and the hub shows it next to every observation.
OBSERVATIONS = {
    "HIGH-VOLUME ADVANCE": ("#3498db", "🔊"), "NEAR 52-WEEK HIGH": ("#f1c40f", "🏔️"),
    "OVERSOLD QUALITY": ("#2ecc71", "💎"), "RSI OVERBOUGHT": ("#e67e22", "🌡️"),
    "WEAK + DOWNTREND": ("#ffa500", "📉"), "HIGH-VOLUME DECLINE": ("#d32f2f", "🔻"),
}


def hot_alerts(prices_full, reco_df, movers_df):
    """Rule-based technical OBSERVATIONS (volume, 52-week high, RSI extremes) on each ticker's latest bar.
    Not signals to act on. Stale tickers produce none; a missing daily move is unknown, never 0; the 52-week high needs
    252 sessions of history (one session no longer makes a "52-week peak")."""
    df_p = prices_full[~prices_full["ticker"].isin(INDICES)]
    latest = df_p.sort_values("date").groupby("ticker").tail(1)
    sessions = df_p.groupby("ticker")["date"].count().rename("sessions")
    hi = (df_p.sort_values("date").groupby("ticker")["price_close"].rolling(window=252, min_periods=252).max()
          .reset_index().groupby("ticker").tail(1).rename(columns={"price_close": "high_52w"}))
    vol = (df_p.sort_values("date").groupby("ticker")["volume"].rolling(window=20, min_periods=20).mean().reset_index()
           .groupby("ticker").tail(1).rename(columns={"volume": "avg_vol_20d"}))
    mv = movers_df[["ticker", "chg_24h", "stale", "last_date"]] if "stale" in movers_df.columns else \
        movers_df[["ticker", "chg_24h"]].assign(stale=False, last_date=pd.NaT)
    a = (reco_df[["ticker", "company", "score", "ma_signal", "rsi"]]
         .merge(latest[["ticker", "price_close", "volume"]], on="ticker")
         .merge(hi[["ticker", "high_52w"]], on="ticker", how="left")
         .merge(vol[["ticker", "avg_vol_20d"]], on="ticker", how="left")
         .merge(sessions, left_on="ticker", right_index=True, how="left")
         .merge(mv, on="ticker", how="left"))
    a = a[~a["stale"].fillna(True).astype(bool)]

    found = []
    for _, r in a.iterrows():
        base = {"ticker": r["ticker"], "name": r["company"],
                "date": None if pd.isna(r.get("last_date")) else pd.Timestamp(r["last_date"]).date()}
        chg = r["chg_24h"] if pd.notna(r["chg_24h"]) else None

        def add(kind, desc):
            colour, icon = OBSERVATIONS[kind]
            found.append({**base, "type": kind, "color": colour, "icon": icon, "desc": desc})
        spike = pd.notna(r["avg_vol_20d"]) and r["avg_vol_20d"] > 0 and r["volume"] > 2 * r["avg_vol_20d"]
        if spike and chg is not None and chg > 0:
            add("HIGH-VOLUME ADVANCE", f"volume {r['volume'] / r['avg_vol_20d'] - 1:+.0%} vs 20-day average, price {chg:+.1f}%")
        if pd.notna(r["high_52w"]) and r["price_close"] >= 0.98 * r["high_52w"]:
            add("NEAR 52-WEEK HIGH", f"close €{r['price_close']:.2f} within 2% of the 52-week high")
        if pd.notna(r["rsi"]) and r["rsi"] < 35 and r["score"] >= ELITE:
            # a screen hit, not a recommendation: only core.decision issues BUY (R3-E)
            add("OVERSOLD QUALITY", f"RSI {r['rsi']:.1f} · Quality {r['score']}")
        if pd.notna(r["rsi"]) and r["rsi"] > 75:
            add("RSI OVERBOUGHT", f"RSI {r['rsi']:.1f} (above 75)")
        if r["score"] < WEAK and r["ma_signal"] in ("BEARISH", "STRONG BEAR"):
            add("WEAK + DOWNTREND", f"Quality {r['score']} · trend {str(r['ma_signal']).lower()}")
        if spike and chg is not None and chg < -3:
            add("HIGH-VOLUME DECLINE", f"volume {r['volume'] / r['avg_vol_20d'] - 1:+.0%} vs 20-day average, price {chg:+.1f}%")
    return found


def breadth_series(prices_full):
    """% of stocks closing above their MA50, per date."""
    b = prices_full[~prices_full["ticker"].isin(BREADTH_EXCLUDE) & prices_full["ma_50"].notna()]
    ts = (b[b["price_close"] > b["ma_50"]].groupby("date")["ticker"].count()
          / b.groupby("date")["ticker"].count() * 100).fillna(0).reset_index()
    ts.columns = ["date", "breadth_pct"]
    return ts


MAX_SCORE = 85       # trend 25 + 5-day momentum 10 + breadth 30 + VIX 10 + macro 10 (market_confidence); NOT 100
REGIMES = (  # (min score, label, colour, read) — a description of the inputs, never an instruction to act
    (75, "STRONG BULLISH", "#2ecc71",
     "Trend, breadth and the volatility / rates inputs are all supportive (heuristic reading, not a forecast)."),
    (50, "BULLISH", "#27ae60",
     "Most inputs are supportive: constructive trend and participation (heuristic reading, not a forecast)."),
    (35, "NEUTRAL / SIDEWAYS", "#f39c12",
     "Mixed or trend-less inputs: neither trend nor breadth is clearly supportive (heuristic reading)."),
    (-1, "BEARISH / CAUTION", "#e74c3c",
     "Weak breadth or a failed trend among the inputs (heuristic reading, not a forecast)."),
)
REGIME_TO_SCORING = {"STRONG BULLISH": "RISK_ON", "BULLISH": "RISK_ON",
                     "NEUTRAL / SIDEWAYS": "NEUTRAL", "BEARISH / CAUTION": "RISK_OFF"}


def _finite(v):
    """float, or None when missing / NaN / inf — a provider series of NaN is unknown, never a level."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def market_confidence(spy, breadth_pct, vix, dxy_5d_move, tnx_chg):
    """0-100 score: SPY trend 25 + SPY 5d momentum 10 + breadth 30 + VIX 10 + macro 10 (max 85).

    `spy` is SPY's price history sorted by date (may be empty). An unknown input (None / NaN) scores the middle of its
    range and is named in the reasons — a transparent imputation, not evidence that the market is neutral.
    """
    vix, dxy_5d_move, tnx_chg, breadth_pct = (_finite(v) for v in (vix, dxy_5d_move, tnx_chg, breadth_pct))
    score, reasons = 0, []
    if not spy.empty:
        last = spy.iloc[-1]
        above50, above200 = last["price_close"] > last["ma_50"], last["price_close"] > last["ma_200"]
        if above50 and above200:
            score += 25
        elif above50 or above200:
            score += 12
            reasons.append("SPY below one MA")
        else:
            reasons.append("SPY below MA50 & MA200")
        ret5 = None
        if len(spy) >= 6:
            ret5 = _finite((float(spy["price_close"].iloc[-1]) / float(spy["price_close"].iloc[-6]) - 1) * 100)
        if ret5 is not None:
            score += int(round(float(np.interp(ret5, [-3, -1, 0, 1.5, 3], [0, 2, 5, 8, 10]))))
            if ret5 < -1:
                reasons.append(f"SPY 5d return {ret5:+.1f}%")
        else:
            score += 5
    # Gradient (no cliff at 50%): 25% = panic (0), 50% = neutral (15), 75%+ = healthy (30)
    if breadth_pct is None:
        score += 15
        reasons.append("Breadth unknown")
    else:
        score += int(round(float(np.interp(breadth_pct, [25, 40, 50, 60, 75], [0, 10, 15, 22, 30]))))
        if breadth_pct < 40:
            reasons.append(f"Breadth Panic ({breadth_pct:.0f}%)")
        elif breadth_pct < 55:
            reasons.append(f"Weak Breadth ({breadth_pct:.0f}%)")
    # An unknown input scores the middle of its range and is named — it is not read as calm (VIX 0) or flat (DXY 0)
    if vix is None:
        score += 5
        reasons.append("VIX unknown")
    else:
        score += int(round(float(np.interp(vix, [15, 20, 25, 30, 40], [10, 8, 4, 1, 0]))))
        if vix > 25:
            reasons.append(f"VIX Risk-Off ({vix:.0f})")
        elif vix > 20:
            reasons.append(f"VIX Elevated ({vix:.0f})")
    if dxy_5d_move is None and tnx_chg is None:
        score += 5
        reasons.append("Macro inputs unknown (DXY / 10Y)")
    elif dxy_5d_move is None or tnx_chg is None:
        # one side known: a known headwind still counts; the unknown side is named
        headwind = (dxy_5d_move is not None and dxy_5d_move >= 1.5) or (tnx_chg is not None and tnx_chg >= 0.12)
        score += 2 if headwind else 5
        reasons.append("DXY unknown" if dxy_5d_move is None else "10Y change unknown")
        if headwind:
            reasons.append(f"Macro Headwind (DXY 5d:{dxy_5d_move:+.1f}%)" if dxy_5d_move is not None
                           else f"Macro Headwind (10Y {tnx_chg:+.2f})")
    elif dxy_5d_move < 0.8 and tnx_chg < 0.08:
        score += 10
    elif dxy_5d_move < 1.5 and tnx_chg < 0.12:
        score += 5
        reasons.append("Mild Macro Friction")
    else:
        reasons.append(f"Macro Headwind (DXY 5d:{dxy_5d_move:+.1f}%)")
    return score, reasons


def regime_from_score(score, tnx_chg=0.0, dxy_pct=0.0, vix=20.0):
    """→ dict(regime, color, advice, scoring_regime) — scoring_regime is the label shown on Market Pulse."""
    for cut, label, colour, advice in REGIMES:
        if score >= cut:
            break
    scoring = REGIME_TO_SCORING[label]
    tnx_chg, dxy_pct, vix = _finite(tnx_chg), _finite(dxy_pct), _finite(vix)
    if None not in (tnx_chg, dxy_pct, vix) and tnx_chg > 0.05 and dxy_pct > 0.1 and vix < 25:  # yields + USD up
        scoring = "INFLATION_SHOCK"
    return {"regime": label, "color": colour, "advice": advice, "scoring_regime": scoring}


def regime_summary(regime: dict, score, reasons, *, breadth=None, vix=None, dxy_pct=None, tnx_chg=None) -> dict:
    """What the market-context answer states, from the ONE regime result (no stance, exposure or tilt of its own):
    {"label", "score", "max_score", "reasons", "incomplete": [inputs that were unknown], "rates_usd_flag": bool}.
    `incomplete` names the inputs scored as unknown (market_confidence gives them the middle of their range)."""
    unknown = [name for name, v in (("VIX", vix), ("DXY", dxy_pct), ("10Y yield change", tnx_chg), ("Breadth", breadth))
               if _finite(v) is None]
    return {"label": regime["regime"], "score": score, "max_score": MAX_SCORE, "reasons": list(reasons or []),
            "incomplete": unknown, "rates_usd_flag": regime.get("scoring_regime") == "INFLATION_SHOCK"}


def dxy_5d_move(prices_full, fallback_pct):
    """Cumulative 5-day % move of DX-Y.NYB; falls back to the live daily % when missing."""
    d = prices_full[prices_full["ticker"] == "DX-Y.NYB"].sort_values("date").tail(6)
    if len(d) >= 2:
        move = _finite((float(d["price_close"].iloc[-1]) / float(d["price_close"].iloc[0]) - 1) * 100)
        if move is not None:
            return move
    return _finite(fallback_pct)


def cap_weighted_quality(reco_df):
    """Market-cap-weighted mean quality score of the stock universe."""
    v = reco_df[~reco_df["ticker"].isin(INDICES)].dropna(subset=["score", "market_cap"])
    if not v.empty and v["market_cap"].sum() > 0:
        return float(np.average(v["score"], weights=v["market_cap"]))
    return float(reco_df[~reco_df["ticker"].isin(INDICES)]["score"].mean())


def upcoming_earnings(earnings_cal, days, today=None):
    today = (today or pd.Timestamp.now()).normalize().date()
    if earnings_cal.empty:
        return earnings_cal
    d = earnings_cal["earnings_date"].dt.date
    return earnings_cal[(d >= today) & (d <= today + timedelta(days=days))]

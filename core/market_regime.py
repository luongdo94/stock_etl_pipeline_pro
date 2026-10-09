"""Market-wide context for the dashboard header — pure functions, no Streamlit.

Horizon window, movers, hot alerts, breadth and the 0-100 market-confidence score / regime.
The regime is a heuristic (not validated against returns); the UI says so.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.rating import QUALITY_TIERS

INDICES = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
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


def movers(prices_full, top=5):
    """Day-over-day % change on the last two dates → (movers_df, gainers, losers)."""
    dates = np.sort(prices_full["date"].unique())
    last = dates[-1]
    prev = dates[-2] if len(dates) > 1 else last
    stocks = prices_full[~prices_full["ticker"].isin(INDICES)]
    m = stocks[stocks["date"] == last].merge(stocks[stocks["date"] == prev][["ticker", "price_close"]],
                                             on="ticker", suffixes=("", "_prev"))
    m["chg_24h"] = (m["price_close"] / m["price_close_prev"] - 1) * 100
    return (m, m.sort_values("chg_24h", ascending=False).head(top),
            m.sort_values("chg_24h", ascending=True).head(top))


def hot_alerts(prices_full, reco_df, movers_df):
    """Rule-based alerts (volume spikes, 52w peaks, RSI extremes) on the latest bar per ticker."""
    df_p = prices_full
    latest = df_p.sort_values("date").groupby("ticker").tail(1)
    hi = (df_p.groupby("ticker")["price_close"].rolling(window=252, min_periods=1).max().reset_index()
          .groupby("ticker").tail(1).rename(columns={"price_close": "high_52w"}))
    vol = (df_p.groupby("ticker")["volume"].rolling(window=20, min_periods=1).mean().reset_index()
           .groupby("ticker").tail(1).rename(columns={"volume": "avg_vol_20d"}))
    a = (reco_df[["ticker", "company", "score", "ma_signal", "rsi"]]
         .merge(latest[["ticker", "price_close", "volume"]], on="ticker")
         .merge(hi[["ticker", "high_52w"]], on="ticker")
         .merge(vol[["ticker", "avg_vol_20d"]], on="ticker"))
    a = a[~a["ticker"].isin(INDICES)].merge(movers_df[["ticker", "chg_24h"]], on="ticker", how="left")
    a["chg_24h"] = a["chg_24h"].fillna(0)

    found = []
    for _, r in a.iterrows():
        base = {"ticker": r["ticker"], "name": r["company"]}
        spike = r["avg_vol_20d"] > 0 and r["volume"] > 2 * r["avg_vol_20d"]
        if spike and r["chg_24h"] > 0:
            found.append({**base, "type": "BULLISH VOL", "color": "#3498db", "icon": "🔊",
                          "desc": f"Vol Spike (+{((r['volume'] / r['avg_vol_20d']) - 1) * 100:.0f}%) | Price ↗"})
        if r["price_close"] >= 0.98 * r["high_52w"]:
            found.append({**base, "type": "52W PEAK", "color": "#f1c40f", "icon": "🏔️",
                          "desc": f"Price: €{r['price_close']:.2f} (Near High)"})
        if r["rsi"] < 35 and r["score"] >= ELITE:
            found.append({**base, "type": "GOLDEN BUY", "color": "#2ecc71", "icon": "💎",
                          "desc": f"RSI: {r['rsi']:.1f} | Score: {r['score']}"})
        if r["rsi"] > 75:
            found.append({**base, "type": "EXIT / RISK", "color": "#ff4b4b", "icon": "",
                          "desc": f"Extreme Overbought (RSI: {r['rsi']:.1f})"})
        if r["score"] < WEAK and r["ma_signal"] in ("BEARISH", "STRONG BEAR"):
            found.append({**base, "type": "BEARISH BLOW", "color": "#ffa500", "icon": "",
                          "desc": "Weak Fundamentals + Bearish Trend"})
        if spike and r["chg_24h"] < -3:
            found.append({**base, "type": "PANIC DUMP", "color": "#d32f2f", "icon": "",
                          "desc": "Heavy Selling | Vol Spike & Price ↘"})
    return found


def breadth_series(prices_full):
    """% of stocks closing above their MA50, per date."""
    b = prices_full[~prices_full["ticker"].isin(BREADTH_EXCLUDE) & prices_full["ma_50"].notna()]
    ts = (b[b["price_close"] > b["ma_50"]].groupby("date")["ticker"].count()
          / b.groupby("date")["ticker"].count() * 100).fillna(0).reset_index()
    ts.columns = ["date", "breadth_pct"]
    return ts


REGIMES = (  # (min score, label, colour, read)
    (75, "STRONG BULLISH", "#2ecc71",
     "Market internals are robust with strong trend alignment. Ideal for aggressive growth deployment."),
    (50, "BULLISH", "#27ae60",
     "Constructive environment. Focus on quality growth and leaders breaking out on volume."),
    (35, "NEUTRAL / SIDEWAYS", "#f39c12",
     "Trend-less environment. Stick to selective bottom-up picking and range-bound strategies."),
    (-1, "BEARISH / CAUTION", "#e74c3c",
     "Defensive posture required. Breadth is deteriorating or trend has failed. Focus on capital preservation."),
)
REGIME_TO_SCORING = {"STRONG BULLISH": "RISK_ON", "BULLISH": "RISK_ON",
                     "NEUTRAL / SIDEWAYS": "NEUTRAL", "BEARISH / CAUTION": "RISK_OFF"}


def market_confidence(spy, breadth_pct, vix, dxy_5d_move, tnx_chg):
    """0-100 score: SPY trend 25 + SPY 5d momentum 10 + breadth 30 + VIX 10 + macro 10 (max 85).

    `spy` is SPY's price history sorted by date (may be empty).
    """
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
        if len(spy) >= 6:
            ret5 = (float(spy["price_close"].iloc[-1]) / float(spy["price_close"].iloc[-6]) - 1) * 100
            score += int(round(float(np.interp(ret5, [-3, -1, 0, 1.5, 3], [0, 2, 5, 8, 10]))))
            if ret5 < -1:
                reasons.append(f"SPY 5d return {ret5:+.1f}%")
        else:
            score += 5
    # Gradient (no cliff at 50%): 25% = panic (0), 50% = neutral (15), 75%+ = healthy (30)
    score += int(round(float(np.interp(breadth_pct, [25, 40, 50, 60, 75], [0, 10, 15, 22, 30]))))
    if breadth_pct < 40:
        reasons.append(f"Breadth Panic ({breadth_pct:.0f}%)")
    elif breadth_pct < 55:
        reasons.append(f"Weak Breadth ({breadth_pct:.0f}%)")
    score += int(round(float(np.interp(vix, [15, 20, 25, 30, 40], [10, 8, 4, 1, 0]))))
    if vix > 25:
        reasons.append(f"VIX Risk-Off ({vix:.0f})")
    elif vix > 20:
        reasons.append(f"VIX Elevated ({vix:.0f})")
    if dxy_5d_move < 0.8 and tnx_chg < 0.08:
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
    if tnx_chg > 0.05 and dxy_pct > 0.1 and vix < 25:  # yields and USD rising together
        scoring = "INFLATION_SHOCK"
    return {"regime": label, "color": colour, "advice": advice, "scoring_regime": scoring}


def dxy_5d_move(prices_full, fallback_pct):
    """Cumulative 5-day % move of DX-Y.NYB; falls back to the live daily % when missing."""
    d = prices_full[prices_full["ticker"] == "DX-Y.NYB"].sort_values("date").tail(6)
    if len(d) >= 2:
        return (float(d["price_close"].iloc[-1]) / float(d["price_close"].iloc[0]) - 1) * 100
    return fallback_pct


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

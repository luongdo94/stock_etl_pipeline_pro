"""Single-ticker TECHNICAL-TIMING backtest (pure). Not evidence for 3-5-year fundamental decisions (audit F-27):
single stock, today's universe, closes only (a stop is detected on a close and filled at the next close, so gaps are
paid but intraday triggers are not modelled), no point-in-time fundamentals, no spread / impact model."""
import numpy as np
import pandas as pd

# The "quality" rule reads TODAY's score on every past bar: a timing demonstration with lookahead, never evidence that
# fundamentals predict returns (audit F31). The label says so wherever the result is shown or exported.
STATIC_QUALITY = "Static Quality demo"


# ── STRATEGY ENGINE ──────────────────────────────────────────────────────────
def run_backtest_simulation(bt_ticker, bt_prices, strategy_type, sl_pct, tp_pct, tx_cost_pct, initial_capital, reco_df):
    """
    Core simulator: Runs a single-ticker backtest for a specific strategy.
    Returns a dict with processed metrics and curves.
    """
    if len(bt_prices) < 60:
        return None

    ticker_score_row = reco_df[reco_df["ticker"] == bt_ticker]
    static_score = int(ticker_score_row["score"].iloc[0]) if not ticker_score_row.empty else 50

    prices_arr  = bt_prices["price_close"].values
    # A single NaN return would turn the whole cumprod equity curve into NaN
    raw_rets = bt_prices["daily_return_pct"].values.astype(float) / 100
    # A gap inside the history is unknown, not 0% (audit F31): it is still filled so the curve can be drawn, but the
    # result carries the count and the view refuses to treat such a run as a clean performance record
    _first = int(np.argmax(~np.isnan(raw_rets))) if (~np.isnan(raw_rets)).any() else len(raw_rets)
    missing_returns = int(np.isnan(raw_rets[_first + 1:]).sum()) if _first < len(raw_rets) else 0
    returns_arr = np.nan_to_num(raw_rets)
    dates_arr   = bt_prices["date"].values
    
    # Fetch indicators
    def _col(name, default):     # a missing column must not crash a strategy that does not use it (audit A11)
        return bt_prices[name].to_numpy(dtype=float) if name in bt_prices.columns else np.full(len(prices_arr), default)
    ma20_arr = _col("ma_20", 0.0)
    ma50_arr = _col("ma_50", 0.0)
    rsi_arr  = _col("rsi", 50.0)

    # Z-Score Calculation
    price_series = pd.Series(prices_arr)
    ma60 = price_series.rolling(60).mean().values
    std60 = price_series.rolling(60).std().values
    z_scores = np.full(len(prices_arr), np.nan, dtype=float)
    for j in range(len(prices_arr)):
        if np.isfinite(std60[j]) and std60[j] > 0:
            z_scores[j] = (prices_arr[j] - ma60[j]) / std60[j]

    is_breakout = "RSI 50 Cross" in strategy_type or "Multi-Indicator Breakout" in strategy_type
    position   = np.zeros(len(bt_prices))
    in_position = False
    entry_price_val = 0.0
    trade_log = []
    
    # Signals are computed from bar i's CLOSE, so they can only be filled at the NEXT bar's close (R3-A: the order used
    # to fill at the very close that produced the signal). A signal on the last bar is not filled.
    n_bars = len(prices_arr)
    for i in range(1, n_bars):
        current_price = prices_arr[i]
        can_fill = i + 1 < n_bars
        fill_price = prices_arr[i + 1] if can_fill else None
        p_date = str(dates_arr[i + 1])[:10] if can_fill else str(dates_arr[i])[:10]
        position[i] = position[i-1]
        exited_today = False

        # 1. Exit Conditions
        if in_position:
            pnl_pct = (current_price - entry_price_val) / entry_price_val
            exit_signal = False
            exit_reason = ""
            
            if pnl_pct <= -sl_pct:
                exit_signal = True; exit_reason = "Stop Loss"
            elif tp_pct > 0 and pnl_pct >= tp_pct:
                exit_signal = True; exit_reason = "Take Profit"
            
            if not exit_signal:
                if "Trend Following" in strategy_type:
                    if ma20_arr[i] < ma50_arr[i]: exit_signal = True; exit_reason = "MA Death Cross"
                elif "RSI Mean Reversion" in strategy_type:
                    if rsi_arr[i] > 70: exit_signal = True; exit_reason = "RSI Overbought"
                elif "Z-Score" in strategy_type:
                    if np.isfinite(z_scores[i]) and z_scores[i] > 0.5: exit_signal = True; exit_reason = "Z-Score > 0.5"
                elif STATIC_QUALITY in strategy_type:
                    if ma20_arr[i] < ma50_arr[i]: exit_signal = True; exit_reason = "Trend Change"
                elif "Buy on Dip" in strategy_type or is_breakout:
                    if current_price < ma50_arr[i]: exit_signal = True; exit_reason = "Price < MA50"
                    
            if exit_signal and can_fill:
                position[i] = 0; in_position = False; exited_today = True
                # realised at the next close: a gap after a stop signal is paid, not assumed away
                net_pnl = (fill_price - entry_price_val) / entry_price_val - 2 * tx_cost_pct
                trade_log.append({"Date": p_date, "Action": "🔴 SELL", "Reason": exit_reason, "Price": f"€{fill_price:.2f}",
                                  "PnL": f"{net_pnl*100:+.1f}%", "_pnl": net_pnl})
        
        # 2. Entry Conditions — never re-enter on the bar we just exited, otherwise level-based
        #    rules (e.g. Institutional Quality) re-buy immediately after a Stop Loss: the stop
        #    is silently undone and the round trip pays no transaction cost.
        if not in_position and not exited_today:
            entry_signal = False; entry_reason = ""
            if "Trend Following" in strategy_type:
                if ma20_arr[i] > ma50_arr[i] and ma20_arr[i-1] <= ma50_arr[i-1]: entry_signal = True; entry_reason = "MA Golden Cross"
            elif "RSI Mean Reversion" in strategy_type:
                if rsi_arr[i] < 30 and rsi_arr[i-1] >= 30: entry_signal = True; entry_reason = "RSI Oversold"
            elif "Z-Score" in strategy_type:
                if (np.isfinite(z_scores[i]) and np.isfinite(z_scores[i-1])
                        and z_scores[i] < -2.0 and z_scores[i-1] >= -2.0):
                    entry_signal = True; entry_reason = "Z-Score < -2.0"
            elif STATIC_QUALITY in strategy_type:
                if static_score >= 75 and ma20_arr[i] > ma50_arr[i]: entry_signal = True; entry_reason = "High Quality + Trend"
            elif "Buy on Dip" in strategy_type:
                if ma20_arr[i] > ma50_arr[i] and rsi_arr[i] < 40 and rsi_arr[i-1] >= 40: entry_signal = True; entry_reason = "Uptrend + RSI Dip"
            elif is_breakout:
                if current_price > ma50_arr[i] and rsi_arr[i] > 50 and rsi_arr[i-1] <= 50: entry_signal = True; entry_reason = "Price>MA50 + RSI crosses 50"
                
            if entry_signal and can_fill:
                position[i] = 1; in_position = True; entry_price_val = fill_price
                trade_log.append({"Date": p_date, "Action": "🟢 BUY", "Reason": entry_reason, "Price": f"€{fill_price:.2f}", "PnL": "-"})

    # position[i] = decision taken at bar i's close; held[t] = position after the fill at bar t's close = position[t-1].
    # The fee is charged on the fill bar (including the LAST bar), the return is earned from the bar after the fill on
    # (audit A11: the old shifted fee dropped a final-bar trade's cost).
    held = np.roll(position, 1); held[0] = 0
    held_prev = np.roll(held, 1); held_prev[0] = 0
    fills = np.abs(np.diff(held, prepend=0.0))
    strategy_returns = returns_arr * held_prev - fills * tx_cost_pct
    cum_strategy = (1 + strategy_returns).cumprod()
    equity_curve = cum_strategy * initial_capital
    
    bnh_returns = returns_arr.copy()
    bnh_returns[0] = 0.0  # Benchmark starts at the first displayed close, not the preceding close.
    cum_bnh = (1 + bnh_returns).cumprod()
    bnh_curve = cum_bnh * initial_capital
    
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    bnh_return = (bnh_curve[-1] / initial_capital - 1) * 100
    # No trades / no variance → Sharpe is undefined (dividing by ~0 used to print -2,519,763)
    _vol = strategy_returns.std()
    sharpe = float("nan") if (not np.any(held) or not np.isfinite(_vol) or _vol < 1e-12) else \
        ((strategy_returns - 0.04 / 252).mean() / _vol) * np.sqrt(252)
    max_dd = ((equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)).min() * 100
    
    # Win rate: computed per completed trade (BUY→SELL pair) from trade_log
    completed_trades = [t["_pnl"] for t in trade_log if t["Action"] == "🔴 SELL"]
    if completed_trades:
        pnl_values = completed_trades                 # unrounded (the display string rounds to 0.1%)
        win_rate = sum(1 for p in pnl_values if p > 0) / len(pnl_values) * 100
    else:
        win_rate = 0.0
    n_trades = len(completed_trades)

    return {
        "ticker": bt_ticker, "strategy": strategy_type,
        # static_score is TODAY's quality score applied to every past bar → lookahead bias.
        # Such results are shown for reference but never crowned the winner.
        "lookahead": STATIC_QUALITY in strategy_type, "missing_returns": missing_returns,
        "total_return": total_return, "bnh_return": bnh_return, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate, "n_trades": n_trades, "n_bars": int(n_bars), "bars_held": int(held_prev.sum()),
        "dates_arr": dates_arr, "equity_curve": equity_curve,
        "bnh_curve": bnh_curve, "trade_log": trade_log
    }

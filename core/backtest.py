"""Single-ticker strategy backtest simulator (pure)."""
import numpy as np
import pandas as pd


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
    returns_arr = np.nan_to_num(bt_prices["daily_return_pct"].values.astype(float) / 100)
    dates_arr   = bt_prices["date"].values
    
    # Fetch indicators
    ma20_arr = bt_prices.get("ma_20", np.zeros_like(prices_arr)).values
    ma50_arr = bt_prices.get("ma_50", np.zeros_like(prices_arr)).values
    rsi_arr  = bt_prices.get("rsi", np.full_like(prices_arr, 50)).values

    # Z-Score Calculation
    price_series = pd.Series(prices_arr)
    ma60 = price_series.rolling(60).mean().values
    std60 = price_series.rolling(60).std().values
    z_scores = np.zeros_like(prices_arr)
    for j in range(len(prices_arr)):
        if std60[j] > 0 and not np.isnan(std60[j]):
            z_scores[j] = (prices_arr[j] - ma60[j]) / std60[j]

    position   = np.zeros(len(bt_prices))
    in_position = False
    entry_price_val = 0.0
    trade_log = []
    
    for i in range(1, len(prices_arr)):
        current_price = prices_arr[i]
        p_date = str(dates_arr[i])[:10]
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
                    if z_scores[i] > 0.5: exit_signal = True; exit_reason = "Z-Score > 0.5"
                elif "Institutional Quality" in strategy_type:
                    if static_score < 60 or ma20_arr[i] < ma50_arr[i]: exit_signal = True; exit_reason = "Quality Degraded/Trend Change"
                elif "Buy on Dip" in strategy_type or "Multi-Indicator Breakout" in strategy_type:
                    if current_price < ma50_arr[i]: exit_signal = True; exit_reason = "Price < MA50"
                    
            if exit_signal:
                position[i] = 0; in_position = False; exited_today = True
                trade_log.append({"Date": p_date, "Action": "🔴 SELL", "Reason": exit_reason, "Price": f"€{current_price:.2f}", "PnL": f"{pnl_pct*100:+.1f}%"})
        
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
                if z_scores[i] < -2.0 and z_scores[i-1] >= -2.0: entry_signal = True; entry_reason = "Z-Score < -2.0"
            elif "Institutional Quality" in strategy_type:
                if static_score >= 75 and ma20_arr[i] > ma50_arr[i]: entry_signal = True; entry_reason = "High Quality + Trend"
            elif "Buy on Dip" in strategy_type:
                if ma20_arr[i] > ma50_arr[i] and rsi_arr[i] < 40 and rsi_arr[i-1] >= 40: entry_signal = True; entry_reason = "Uptrend + RSI Dip"
            elif "Multi-Indicator Breakout" in strategy_type:
                if current_price > ma50_arr[i] and rsi_arr[i] > 50 and rsi_arr[i-1] <= 50: entry_signal = True; entry_reason = "Price>MA50 + RSI>50"
                
            if entry_signal:
                position[i] = 1; in_position = True; entry_price_val = current_price
                trade_log.append({"Date": p_date, "Action": "🟢 BUY", "Reason": entry_reason, "Price": f"€{current_price:.2f}", "PnL": "-"})

    pos_shifted = np.roll(position, 1); pos_shifted[0] = 0
    signal_changes = np.abs(np.diff(pos_shifted, prepend=pos_shifted[0]))
    strategy_returns = returns_arr * pos_shifted - signal_changes * tx_cost_pct
    cum_strategy = (1 + strategy_returns).cumprod()
    equity_curve = cum_strategy * initial_capital
    
    cum_bnh = (1 + returns_arr).cumprod()
    bnh_curve = cum_bnh * initial_capital
    
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    bnh_return = (bnh_curve[-1] / initial_capital - 1) * 100
    # No trades / no variance → Sharpe is undefined (dividing by ~0 used to print -2,519,763)
    _vol = strategy_returns.std()
    sharpe = float("nan") if (not np.any(position) or not np.isfinite(_vol) or _vol < 1e-12) else \
        ((strategy_returns - 0.04 / 252).mean() / _vol) * np.sqrt(252)
    max_dd = ((equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)).min() * 100
    
    # Win rate: computed per completed trade (BUY→SELL pair) from trade_log
    completed_trades = [(t["PnL"]) for t in trade_log if t["Action"] == "🔴 SELL"]
    if completed_trades:
        # Parse "+3.5%" → 3.5, "-2.1%" → -2.1
        pnl_values = [float(p.replace("%", "")) for p in completed_trades]
        win_rate = sum(1 for p in pnl_values if p > 0) / len(pnl_values) * 100
    else:
        win_rate = 0.0
    n_trades = len(completed_trades)

    return {
        "ticker": bt_ticker, "strategy": strategy_type,
        # static_score is TODAY's quality score applied to every past bar → lookahead bias.
        # Such results are shown for reference but never crowned the winner.
        "lookahead": "Institutional Quality" in strategy_type,
        "total_return": total_return, "bnh_return": bnh_return, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate, "n_trades": n_trades, "dates_arr": dates_arr, "equity_curve": equity_curve,
        "bnh_curve": bnh_curve, "trade_log": trade_log
    }

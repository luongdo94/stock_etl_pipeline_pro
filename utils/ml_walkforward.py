"""
Walk-forward experiment: do the ML Predictor's neural engines beat a no-change forecast on real stocks?

    python utils/ml_walkforward.py --tickers 20 --horizon 21 --windows 4 --spacing 63 --out logs/ml_walkforward.csv

For each sampled ticker and each of `windows` evaluation windows (ends spaced `spacing` trading days apart, most
recent last) the engines are trained ONLY on data before the window and asked for the next `horizon` closes:

  naive        price stays at the last known close
  drift        last close grown at the mean daily log return of the previous 120 days
  arima        pmdarima auto_arima on the last 500 closes
  lstm / transformer / patchtst   the architectures of views/ml_predictor.py (same inputs, residual-to-last-price output,
               Huber loss, 30 epochs, fixed seed). Inputs: price, return, SPY and VIX returns, volume surge, RSI, Z-score,
               OBV rate of change (the market-regime score of the app is left out: it needs the breadth series).

Metrics per forecast: RMSE of the price path, skill = 1 - RMSE/RMSE(naive), error of the final price, direction hit.
The windows of different tickers overlap in calendar time and move together, so the number of independent
observations is smaller than the number of rows; read the result with that in mind.
"""
import argparse
import logging
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from core import ml_forecast as mlf  # noqa: E402


class StockLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=21):
        super().__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        attn, _ = self.attention(out, out, out)
        return self.fc(attn.mean(dim=1))


class StockTransformer(torch.nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=21, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_size, d_model)
        self.pos_enc = torch.nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                 dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.fc = torch.nn.Linear(d_model, output_size)

    def forward(self, x):
        T = x.shape[1]
        x = self.input_proj(x) + self.pos_enc[:, :T, :]
        return self.fc(self.norm(self.encoder(x)[:, -1, :]))


class StockPatchTST(torch.nn.Module):
    def __init__(self, context_window, target_window, patch_len=16, stride=8, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.num_patches = (context_window - patch_len) // stride + 1
        self.patch_embed = torch.nn.Linear(patch_len, d_model)
        self.pos_enc = torch.nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                 dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(self.num_patches * d_model, target_window)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1).unfold(-1, self.patch_len, self.stride)
        P = x.shape[2]
        x = self.norm(self.encoder(self.patch_embed(x.reshape(B * C, P, self.patch_len)) + self.pos_enc[:, :P, :]))
        return self.head(x.reshape(B, C, P * x.shape[-1]).mean(dim=1))


def build_features(df: pd.DataFrame, spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    d = d.merge(spy, on="date", how="left").merge(vix, on="date", how="left")
    d["spy_ret"], d["vix_ret"] = d["spy_ret"].fillna(0), d["vix_ret"].fillna(0)
    d["vol_surge"] = d["volume"] / d["volume"].rolling(20).mean().fillna(d["volume"])
    d["rsi"] = d["rsi"].fillna(50.0) if "rsi" in d.columns else 50.0
    d["price_z_score"] = d["price_z_score"].fillna(0.0) if "price_z_score" in d.columns else 0.0
    obv = (np.sign(d["price_close"].diff().fillna(0)) * d["volume"]).cumsum()
    d["obv_roc"] = obv.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
    cols = ["price_close", "daily_return_pct", "spy_ret", "vix_ret", "vol_surge", "rsi", "price_z_score", "obv_roc"]
    d[cols] = d[cols].ffill().fillna(0)
    return d[["date"] + cols]


def nn_forecast(kind, feats: pd.DataFrame, lookback: int, horizon: int, epochs: int, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = feats.drop(columns="date").to_numpy(dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
    price_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data[:, 0:1])
    sc = scaler.transform(data)
    starts = range(lookback, len(sc) - horizon)
    if len(starts) < 30:
        return None
    X = torch.FloatTensor(np.array([sc[i - lookback:i] for i in starts]))
    y = torch.FloatTensor(np.array([sc[i:i + horizon, 0] for i in starts]))
    n_feat = sc.shape[1]
    if kind == "lstm":
        model, lr, crit = StockLSTM(n_feat, output_size=horizon), 1e-3, torch.nn.HuberLoss(delta=1.0)
    elif kind == "transformer":
        model, lr, crit = StockTransformer(n_feat, output_size=horizon), 1e-3, torch.nn.HuberLoss(delta=0.5)
    else:
        model, lr, crit = StockPatchTST(lookback, horizon), 8e-4, torch.nn.HuberLoss(delta=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(X.size(0))
        for s in range(0, X.size(0), 128):
            b = idx[s:s + 128]
            opt.zero_grad()
            loss = crit(model(X[b]) + X[b][:, -1, 0].unsqueeze(1), y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        last = torch.FloatTensor(sc[-lookback:]).unsqueeze(0)
        pred = (model(last) + last[:, -1, 0].unsqueeze(1)).numpy().flatten()
    return price_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()


def arima_forecast(closes: np.ndarray, horizon: int):
    try:
        from pmdarima import auto_arima
        return np.asarray(auto_arima(closes, seasonal=False, stepwise=True, suppress_warnings=True).predict(n_periods=horizon))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--spacing", type=int, default=63)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", default="lstm,transformer,patchtst,arima")
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles every n-th sampled ticker (run n processes in parallel)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads per process (0 = library default)")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "ml_walkforward.csv"))
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    from services.db import load_data
    prices = load_data()[0]
    spy = prices[prices.ticker == "SPY"][["date", "daily_return_pct"]].rename(columns={"daily_return_pct": "spy_ret"})
    vix = prices[prices.ticker == "^VIX"][["date", "daily_return_pct"]].rename(columns={"daily_return_pct": "vix_ret"})
    skip = {"SPY", "^VIX", "^GSPC", "^DJI", "^IXIC"}
    eligible = [t for t, g in prices.groupby("ticker") if t not in skip and len(g) >= 700]
    rng = np.random.default_rng(a.seed)
    sample = sorted(rng.choice(eligible, min(a.tickers, len(eligible)), replace=False))
    i, nshards = (int(x) for x in a.shard.split("/"))
    sample = sample[i::nshards]
    models = a.models.split(",")
    lookback = 90 if a.horizon <= 14 else 180
    rows, t0 = [], time.time()
    for k, t in enumerate(sample, 1):
        feats_all = build_features(prices[prices.ticker == t].tail(900), spy, vix)
        n = len(feats_all)
        for w in range(a.windows):
            end = n - (a.windows - 1 - w) * a.spacing                # most recent window last
            start = end - a.horizon
            if start < lookback + a.horizon + 60:
                continue
            train = feats_all.iloc[:start].tail(500).reset_index(drop=True)
            actual = feats_all["price_close"].to_numpy()[start:end]
            last = float(train["price_close"].iloc[-1])
            fc = {"naive": np.full(a.horizon, last)}
            mu = float(np.log(train["price_close"]).diff().tail(120).mean())
            fc["drift"] = last * np.exp(mu * np.arange(1, a.horizon + 1))
            for m in models:
                p = arima_forecast(train["price_close"].to_numpy(), a.horizon) if m == "arima" \
                    else nn_forecast(m, train, lookback, a.horizon, a.epochs)
                if p is not None:
                    fc[m] = np.asarray(p)[:a.horizon]
            for name, path in fc.items():
                r = mlf.skill_vs_naive(actual, path, last)
                rows.append({"ticker": t, "window": w, "end_date": str(feats_all["date"].iloc[end - 1])[:10], "model": name,
                             "rmse": r["rmse"], "naive_rmse": r["naive_rmse"], "skill": r["skill"],
                             "final_err_pct": r["final_err_pct"], "dir_hit": r["dir_hit"],
                             "pred_ret": float(path[-1] / last - 1), "actual_ret": float(actual[-1] / last - 1)})
        print(f"[{k}/{len(sample)}] {t}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("done", a.out)


if __name__ == "__main__":
    main()

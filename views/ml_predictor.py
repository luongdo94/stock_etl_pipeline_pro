"""View: 🤖 ML Predictor"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.smart_money import get_sm_spirit_unified_v2
from etl.utils import compute_score
from services.ai import analyze_sentiment_finbert
from ui.icons import SVG_ICONS, render_header
import numpy as np


def render(ctx):
    """Render the 🤖 ML Predictor tab. ctx is the app globals() dict."""
    breadth_ts_global = ctx['breadth_ts_global']
    companies_full = ctx['companies_full']
    current_universe = ctx['current_universe']
    df_spy_global = ctx['df_spy_global']
    format_ticker = ctx['format_ticker']
    prices_full = ctx['prices_full']
    regime = ctx['regime']
    regime_ui_color = ctx['regime_ui_color']
    import torch
    import optuna
    pass  # hoisted to module level: import numpy as np
    from arch import arch_model
    render_header("zap", "Context-Aware Direct Multi-Step Forecasting (v11.0)", "Institutional-Grade Adaptive ML Ensemble")
    st.warning("🧪 Experimental: these forecasts have not been shown to beat a no-change forecast out of "
               "sample. Use them for exploration only — the Decision Summary in Stock Analysis ignores them.")
    
    # ── XGBoost BUY/SELL Classifier (Scale-Invariant Signal) ─────────────────────
    @st.cache_data(show_spinner="🌲 XGBoost: Training Directional Signal Classifier...")
    def run_xgboost_signal(df_ticker, horizon_days: int = 5):
        """
        XGBoost binary classifier predicting price direction (BUY/SELL/NEUTRAL).
        Operates entirely on returns and technical ratios — immune to price scale.
        
        Features: lag returns (1,2,5d), rolling volatility (10,21d), RSI approx,
                  MACD approx, Bollinger width, volume change.
        Target  : 1 (BUY) if forward_return > +0.5%, -1 (SELL) if < -0.5%, else 0.
        Returns : (signal_label, probability, feature_importance_dict)
        """
        try:
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import TimeSeriesSplit

            df = df_ticker.copy().sort_values("date").reset_index(drop=True)
            if len(df) < 120:
                return "NEUTRAL", 0.5, {}

            c = df["price_close"].values
            v = df["volume"].values if "volume" in df.columns else np.ones(len(c))

            # ── Feature Engineering ──────────────────────────────────────────────
            ret1  = np.diff(c, prepend=c[0]) / (np.abs(c) + 1e-8)
            ret2  = np.concatenate([[0, 0], (c[2:] - c[:-2]) / (np.abs(c[:-2]) + 1e-8)])
            ret5  = np.concatenate([[0]*5, (c[5:] - c[:-5]) / (np.abs(c[:-5]) + 1e-8)])

            def _rolling(arr, w, fn):
                out = np.full(len(arr), np.nan)
                for i in range(w - 1, len(arr)):
                    out[i] = fn(arr[i-w+1:i+1])
                return out

            vol10 = _rolling(ret1, 10, np.std)
            vol21 = _rolling(ret1, 21, np.std)
            ma10  = _rolling(c, 10, np.mean)
            ma21  = _rolling(c, 21, np.mean)
            ma50  = _rolling(c, 50, np.mean)
            # RSI approx
            up   = np.where(ret1 > 0, ret1, 0)
            dn   = np.where(ret1 < 0, -ret1, 0)
            avg_up14 = _rolling(up, 14, np.mean)
            avg_dn14 = _rolling(dn, 14, np.mean)
            rsi  = 100 - 100 / (1 + avg_up14 / (avg_dn14 + 1e-8))
            # MACD signal approx
            macd = (ma10 - ma21) / (np.abs(ma21) + 1e-8) * 100
            # Bollinger width
            std21 = _rolling(c, 21, np.std)
            bb_width = (2 * std21) / (np.abs(ma21) + 1e-8) * 100
            # Volume change
            vol_ch = np.diff(v, prepend=v[0]) / (np.abs(v) + 1e-8)
            # Trend: price vs MA50
            vs_ma50 = (c - ma50) / (np.abs(ma50) + 1e-8) * 100

            X_raw = np.column_stack([
                ret1, ret2, ret5, vol10, vol21,
                rsi, macd, bb_width, vol_ch, vs_ma50
            ])
            feat_names = [
                "ret1", "ret2", "ret5", "vol10", "vol21",
                "rsi", "macd", "bb_width", "vol_ch", "vs_ma50"
            ]

            # ── Target: forward return over horizon_days ─────────────────────────
            fwd_ret = np.concatenate([
                (c[horizon_days:] - c[:-horizon_days]) / (np.abs(c[:-horizon_days]) + 1e-8),
                np.full(horizon_days, np.nan)
            ])
            y_raw = np.where(fwd_ret > 0.005, 1, np.where(fwd_ret < -0.005, -1, 0))

            # Drop NaN rows
            valid = ~(np.isnan(X_raw).any(axis=1) | np.isnan(fwd_ret))
            X, y = X_raw[valid], y_raw[valid]
            if len(X) < 60:
                return "NEUTRAL", 0.5, {}

            # ── Time-Series Train/Test Split (no leakage) ──────────────────────
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:-horizon_days]  # exclude last horizon_days
            y_train, y_test = y[:split], y[split:-horizon_days]

            # ── XGBoost Classifier ──────────────────────────────────────────────
            clf = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss",
                verbosity=0, tree_method="hist"
            )
            # Remap labels: -1→0, 0→1, 1→2 for XGBoost multi-class
            le = LabelEncoder()
            clf.fit(X_train, le.fit_transform(y_train))

            # ── Predict on latest window ────────────────────────────────────────
            last_x = X_raw[-1:].copy()
            if np.isnan(last_x).any():
                return "NEUTRAL", 0.5, {}
            proba = clf.predict_proba(last_x)[0]
            pred_class_idx = int(np.argmax(proba))
            pred_class = le.inverse_transform([pred_class_idx])[0]
            confidence = float(proba[pred_class_idx])

            # ── Feature Importance ──────────────────────────────────────────────
            imp = dict(zip(feat_names, clf.feature_importances_.tolist()))
            imp = {k: round(v * 100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}

            label_map = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}
            return label_map.get(int(pred_class), "NEUTRAL"), confidence, imp

        except Exception as e:
            return "NEUTRAL", 0.5, {}

    # ── ML Model Architectures (Support for 13th Feature: Market Regime) ──────────────
    
    # ── LSTM Architecture (v7.0: Direct Multi-step Mapping) ───────────
    class StockLSTM(torch.nn.Module):
        def __init__(self, input_size=13, hidden_size=64, num_layers=2, output_size=30):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers  = num_layers
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
            self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            # Temporal Attention
            attn_output, _ = self.attention(out, out, out)
            return self.fc(attn_output.mean(dim=1))

    # ── Transformer Architecture (v8.0: Pure Attention — Parallel Multi-step) ────────
    class StockTransformer(torch.nn.Module):
        def __init__(self, input_size=13, d_model=64, nhead=4, num_layers=2, output_size=30, dropout=0.1):
            super().__init__()
            self.input_proj = torch.nn.Linear(input_size, d_model)
            self.pos_enc = torch.nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = torch.nn.LayerNorm(d_model)
            self.fc   = torch.nn.Linear(d_model, output_size)

        def forward(self, x):
            B, T, _ = x.shape
            x = self.input_proj(x)
            x = x + self.pos_enc[:, :T, :]
            x = self.encoder(x)
            x = self.norm(x[:, -1, :])
            return self.fc(x)

    # ── PatchTST Architecture (v10.0: Channel-Independent Transformer) ────────
    class StockPatchTST(torch.nn.Module):
        def __init__(self, c_in=13, context_window=120, target_window=30,
                     patch_len=16, stride=8, d_model=64, nhead=4,
                     num_layers=2, dropout=0.1):
            super().__init__()
            self.patch_len = patch_len
            self.stride    = stride
            self.c_in      = c_in
            self.num_patches = (context_window - patch_len) // stride + 1
            self.patch_embed = torch.nn.Linear(patch_len, d_model)
            self.pos_enc     = torch.nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
            encoder_layer    = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.encoder     = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm        = torch.nn.LayerNorm(d_model)
            self.head        = torch.nn.Linear(self.num_patches * d_model, target_window)

        def forward(self, x):
            B, T, C = x.shape
            x = x.permute(0, 2, 1)                      
            x = x.unfold(-1, self.patch_len, self.stride) 
            P = x.shape[2]
            x = x.reshape(B * C, P, self.patch_len)
            x = self.patch_embed(x)                  
            x = x + self.pos_enc[:, :P, :]            
            x = self.encoder(x)                       
            x = self.norm(x)                          
            x = x.reshape(B, C, P * x.shape[-1])     
            x = x.mean(dim=1)                         
            return self.head(x)

    def _get_regime_history():
        """
        Computes historical 0-100 Market Regime Score for the last 500 days.
        Used as the 13th feature for context-aware forecasting.
        """
        try:
            # 1. Price Trend (50 pts)
            spy = df_spy_global.tail(500).copy()
            spy['trend_score'] = (spy['price_close'] > spy['ma_50']).astype(int) * 25
            spy['trend_score'] += (spy['price_close'] > spy['ma_200']).astype(int) * 25
            
            # 2. Breadth (30 pts)
            br = breadth_ts_global.copy()
            
            # 3. Volatility (20 pts)
            vix_h = prices_full[prices_full['ticker'] == '^VIX'].sort_values('date').tail(500).copy()
            vix_h['vix_score'] = vix_h['price_close'].apply(lambda v: 20 if v < 20 else 10 if v < 30 else 0)
            
            # Sync all on date
            reg_df = spy[['date', 'trend_score']].merge(br, on='date', how='left').merge(vix_h[['date', 'vix_score']], on='date', how='left')
            reg_df['breadth_score'] = (reg_df['breadth_pct'] / 100 * 30).fillna(15)
            reg_df['regime_score'] = reg_df['trend_score'] + reg_df['breadth_score'] + reg_df['vix_score']
            return reg_df[['date', 'regime_score']].fillna(50)
        except Exception:
            return pd.DataFrame()

    def _precompute_features(df_ticker):
        """
        Shared 13-factor feature engineering (Context-Aware v11.0).
        Injected 'Market Regime Score' as the 13th strategic input.

        Returns a dict with:
            data_scaled  : np.ndarray [N, 12]
            price_scaler : MinMaxScaler fitted on raw price_close column
            features     : list[str] of 12 feature names
            data         : np.ndarray [N, 12] (raw, unscaled)
            df           : pd.DataFrame with all features
            n_feat       : int (12)
        Returns None on failure.
        """
        import warnings; warnings.filterwarnings('ignore')
        try:
            ticker_id = df_ticker['ticker'].iloc[0] if not df_ticker.empty else None
            if ticker_id is None:
                return None

            # Cache key includes date to invalidate when data refreshes
            max_date  = str(df_ticker['date'].max()) if 'date' in df_ticker.columns else ''
            cache_key = f"feat_cache_{ticker_id}_{max_date}"
            if cache_key in st.session_state:
                return st.session_state[cache_key]

            df = df_ticker.copy().sort_values("date").reset_index(drop=True).tail(500).reset_index(drop=True)

            # ── Macro & Volatility ──
            df['vol_surge'] = df['volume'] / (df['volume'].rolling(20).mean().fillna(df['volume']))
            spy_df = prices_full[prices_full['ticker']=='SPY'][['date','daily_return_pct']].rename(columns={'daily_return_pct':'spy_ret'})
            vix_df = prices_full[prices_full['ticker']=='^VIX'][['date','daily_return_pct']].rename(columns={'daily_return_pct':'vix_ret'})
            df = df.merge(spy_df, on='date', how='left').merge(vix_df, on='date', how='left')
            df['spy_ret'] = df['spy_ret'].fillna(0)
            df['vix_ret'] = df['vix_ret'].fillna(0)

            # ── Technical ──
            if 'rsi' in df.columns:           df['rsi'] = df['rsi'].fillna(50.0)
            else:                              df['rsi'] = 50.0
            if 'price_z_score' in df.columns: df['price_z_score'] = df['price_z_score'].fillna(0.0)
            else:                              df['price_z_score'] = 0.0

            # ── Fundamentals ──
            co_row = companies_full[companies_full['ticker'] == ticker_id].iloc[0].to_dict() \
                if not companies_full[companies_full['ticker'] == ticker_id].empty else {}
            _ebitda = float(co_row.get('ebitda', 1) or 1)
            _debt   = float(co_row.get('total_debt', 0) or 0)
            df['pe_ratio']    = float(np.clip(float(co_row.get('pe_ratio', 20) or 20), 0, 150))
            df['roe']         = float(np.clip(float(co_row.get('roe', 0) or 0) * 100, -50, 100))
            df['fcf_margin']  = float(np.clip(float(co_row.get('fcf_margin', 0) or 0), -50, 80))
            df['debt_ebitda'] = float(np.clip(_debt / max(_ebitda, 1), 0, 12))
            df['rev_growth']  = float(np.clip(float(co_row.get('revenue_growth', 0) or 0) * 100, -50, 100))

            # ── Market Regime Overlay (13th Feature) ──
            rh = _get_regime_history()
            if not rh.empty:
                df = df.merge(rh, on='date', how='left')
                df['regime_score'] = df['regime_score'].ffill().fillna(50)
            else:
                df['regime_score'] = 50.0

            # ── Smart Money / Volume Dynamics (14th Feature) ──
            # On-Balance Volume Rate of Change (Stationary indicator for Smart Money accumulation)
            _raw_obv = (np.sign(df['price_close'].diff().fillna(0)) * df['volume']).cumsum()
            df['obv_roc'] = _raw_obv.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)
            # Clip extreme values to prevent exploding gradients in ML
            df['obv_roc'] = df['obv_roc'].clip(-5.0, 5.0)

            features = [
                'price_close', 'daily_return_pct',
                'spy_ret', 'vix_ret',
                'vol_surge', 'rsi', 'price_z_score',
                'pe_ratio', 'roe', 'fcf_margin',
                'debt_ebitda', 'rev_growth', 'regime_score',
                'obv_roc'
            ]
            data = df[features].ffill().fillna(0).values.astype(np.float32)

            from sklearn.preprocessing import MinMaxScaler
            scaler       = MinMaxScaler(feature_range=(-1, 1))
            data_scaled  = scaler.fit_transform(data)
            price_scaler = MinMaxScaler(feature_range=(-1, 1))
            price_scaler.fit(data[:, 0:1])

            result = {
                'data_scaled' : data_scaled,
                'price_scaler': price_scaler,
                'features'    : features,
                'data'        : data,
                'df'          : df,
                'n_feat'      : len(features),
            }
            st.session_state[cache_key] = result
            return result
        except Exception:
            return None

    def _run_lstm_core(df_ticker, lookback=60, forecast_days=30, sector_name=None, quality_score=50):
        import warnings
        warnings.filterwarnings('ignore')
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # ── Shared Feature Engineering (cached per ticker) ──
        feat = _precompute_features(df_ticker)
        if feat is None:
            return None, None, None
        data_scaled  = feat['data_scaled']
        price_scaler = feat['price_scaler']
        features     = feat['features']
        data         = feat['data']
        df           = feat['df']

        # Adaptive Lookback tuning (v7.0)
        ticker_vol = df['daily_return_pct'].tail(60).std()
        spy_vol_s  = prices_full[prices_full['ticker']=='SPY']['daily_return_pct'].tail(60)
        spy_vol    = spy_vol_s.std() if not spy_vol_s.empty else 1.0

        # 🛡️ ADAPTIVE CLIPPING & WEIGHTS (v7.0)
        vol_ratio = ticker_vol / (spy_vol + 1e-6)
        dynamic_clamp = 0.05 + min(0.05, 0.02 * vol_ratio)

        if ticker_vol > 2.0 * spy_vol:   lstm_w, arima_w, lookback = 0.70, 0.30, max(60, min(lookback, 60))
        elif ticker_vol < 0.8 * spy_vol: lstm_w, arima_w, lookback = 0.40, 0.60, 120
        else:                             lstm_w, arima_w, lookback = 0.60, 0.40, max(90, lookback)

        if len(data_scaled) < lookback + forecast_days + 30:
            return None, None, None

        X, y = [], []
        for i in range(len(data_scaled) - lookback - forecast_days):
            X.append(data_scaled[i:(i+lookback), :])
            y.append(data_scaled[i+lookback : i+lookback+forecast_days, 0])
        
        X_t = torch.FloatTensor(np.array(X)).to(device)
        y_t = torch.FloatTensor(np.array(y)).to(device)
        
        # 🛡️ TEMPORAL FEATURE DECAY (v7.3: Consistent train/inference)
        # Create decay function to ensure same transformation in training and inference
        def apply_temporal_decay(X_tensor, lookback_len, device):
            """Apply exponential temporal decay: recent data gets higher weight"""
            decay = torch.exp(torch.linspace(-0.5, 0, lookback_len)).to(device).view(1, lookback_len, 1)
            return X_tensor * decay
        
        X_t = apply_temporal_decay(X_t, lookback, device)
        
        ticker_id    = df_ticker['ticker'].iloc[0] if not df_ticker.empty else "unknown"
        MODEL_VERSION = f"v7_direct_{forecast_days}"
        if "optuna_cache" not in st.session_state or st.session_state.get("optuna_version") != MODEL_VERSION:
            st.session_state.optuna_cache = {}; st.session_state.optuna_version = MODEL_VERSION
            
        # FIX: Cache key must include forecast_days and lookback to avoid collision
        cache_key = f"{ticker_id}_{forecast_days}_{lookback}"
        if cache_key in st.session_state.optuna_cache:
            best = st.session_state.optuna_cache[cache_key]
        else:
            hpo_split = int(len(X_t)*0.8)
            X_hpo, y_hpo = X_t[:hpo_split], y_t[:hpo_split]
            def objective(trial):
                h  = trial.suggest_categorical("hidden_size",[32,64,128])
                nl = trial.suggest_int("num_layers",1,2)
                lr = trial.suggest_float("lr",5e-4,2e-3,log=True)
                m  = StockLSTM(input_size=len(features),hidden_size=h,num_layers=nl,output_size=forecast_days).to(device)
                cr = torch.nn.HuberLoss(delta=1.0)
                op = torch.optim.Adam(m.parameters(),lr=lr)
                m.train()
                import time
                for _ in range(20): # Optimized: 20 epochs for HPO
                    indices = torch.randperm(X_hpo.size(0), device=device)
                    for start_idx in range(0, X_hpo.size(0), 128):
                        idx = indices[start_idx:start_idx+128]
                        X_b, y_b = X_hpo[idx], y_hpo[idx]
                        
                        y_baseline = X_b[:, -1, 0].unsqueeze(1) # Anchor
                        op.zero_grad()
                        o = m(X_b) + y_baseline
                        l = cr(o, y_b)
                        l.backward()
                        op.step()
                    time.sleep(0.005) # Micro-yield
                return l.item()
            with st.spinner(f"Tuning Direct Intelligence for {ticker_id}..."):
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=5, timeout=10) # Optimized: 5 trials, 10s
                best = study.best_params; best['epochs']=80
                st.session_state.optuna_cache[cache_key] = best
        
        # ── Final Training (v7.2: Direct Multi-step Architecture) ──
        model = StockLSTM(input_size=len(features), hidden_size=best['hidden_size'], num_layers=best['num_layers'], output_size=forecast_days).to(device)
        cr = torch.nn.HuberLoss(delta=1.0); op = torch.optim.Adam(model.parameters(), lr=best['lr'])
        model.train(); prev_loss = 1e9
        
        for epoch in range(best['epochs']):
            indices = torch.randperm(X_t.size(0), device=device)
            for start_idx in range(0, X_t.size(0), 128):
                idx = indices[start_idx:start_idx+128]
                X_b, y_b = X_t[idx], y_t[idx]
                
                y_baseline = X_b[:, -1, 0].unsqueeze(1)
                op.zero_grad()
                o = model(X_b) + y_baseline
                l_core = cr(o, y_b)
                
                # Multi-step Directional Penalty (v7.3: Increased weight 0.5 → 1.5)
                pred_diff = o - y_baseline
                true_diff = y_b - y_baseline
                penalty = torch.mean(torch.clamp(-pred_diff * true_diff, min=0)) * 1.5
                
                l = l_core + penalty
                l.backward()
                op.step()
            
            if torch.isnan(l): break
            l_val = l.item()
            if abs(prev_loss - l_val) < (prev_loss * 5e-5) and epoch > 30: break
            prev_loss = l_val
            import time; time.sleep(0.005) # Micro-yield
            
        # ── INFERENCE (v7.3: Single Shot Direct with Consistent Decay) ──
        model.eval()
        last_seq = data_scaled[-lookback:].copy()
        last_seq_t = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
        last_seq_t = apply_temporal_decay(last_seq_t, lookback, device)  # Same decay as training
        with torch.no_grad():
            y_base_inf   = last_seq_t[:, -1, 0].unsqueeze(1)
            preds_scaled = (model(last_seq_t) + y_base_inf).cpu().numpy().flatten()
        
        lstm_predicted_prices = price_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        
        # ── Raw ARIMA ──
        ts_raw = df['price_close'].values
        try:
            from pmdarima import auto_arima
            arima_predicted_prices = auto_arima(ts_raw, seasonal=False, stepwise=True, suppress_warnings=True).predict(n_periods=forecast_days)
        except Exception:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                arima_predicted_prices = ARIMA(ts_raw,order=(1,1,1)).fit().forecast(steps=forecast_days)
            except Exception:
                arima_predicted_prices = np.full(forecast_days,ts_raw[-1])
        
        ensemble_prices = (lstm_w * lstm_predicted_prices) + (arima_w * arima_predicted_prices)
        
        # 🛡️ ADAPTIVE VOLATILITY CLIPPING (v7.2)
        clamped_prices = [df_ticker['price_close'].iloc[-1]]
        for t in range(len(ensemble_prices)):
            p_raw = ensemble_prices[t]
            p_prev = clamped_prices[-1]
            p_clamped = np.clip(p_raw, p_prev * (1 - dynamic_clamp), p_prev * (1 + dynamic_clamp))
            clamped_prices.append(p_clamped)
            
        current_price = data[-1,0]
        if np.isnan(ensemble_prices[-1]) or current_price==0: return None,None,None
        
        model.eval()
        X_explain = last_seq_t.clone().requires_grad_(True)
        out_explain = model(X_explain)
        torch.sum(out_explain).backward() # Backprop through entire multi-step output
        importances = torch.abs(X_explain.grad[0]).mean(dim=0).cpu().numpy()
        importances = importances / (np.sum(importances) + 1e-9) * 100
        feat_imp_dict = dict(zip(features, importances))

        return clamped_prices[1:], (clamped_prices[-1]-clamped_prices[0])/clamped_prices[0], feat_imp_dict

    def calculate_backtest_accuracy(df_full, sector_name=None, quality_score=50, test_size=21):
        """Phase 10: Honest Backtest - Strict Train/Test Separation"""
        if len(df_full) < 150: return None, None, None
        # We slice raw data to ensure NO LEAKAGE from the future
        train_df = df_full.iloc[:-test_size].copy()
        actual_prices = df_full["price_close"].iloc[-test_size:].values
        
        # Run forecast strictly on training data
        # No re-training or HPO on the test window allowed
        predicted,_,_ = _run_lstm_core(train_df, lookback=120, forecast_days=test_size, sector_name=sector_name, quality_score=quality_score)
        
        if predicted is None or len(predicted) < test_size: return None, None, None
        mape = np.mean(np.abs((actual_prices - np.asarray(predicted)) / actual_prices))
        # Baseline: "price stays where it is". 100·(1−MAPE) looks like ~95% "precision" for ANY
        # model on a 2-4 week horizon, so the only meaningful question is whether we beat this.
        naive_mape = np.mean(np.abs((actual_prices - train_df["price_close"].iloc[-1]) / actual_prices))
        return max(0.0, min(100.0, 100*(1-mape))), float(mape), float(naive_mape)

    @st.cache_data(show_spinner="Training Adaptive AI Ensemble (LSTM + ARIMA)...")
    def train_predict_lstm(df_ticker, lookback=60, forecast_days=30, sector_name=None, quality_score=50):
        return _run_lstm_core(df_ticker, lookback=lookback, forecast_days=forecast_days, sector_name=sector_name, quality_score=quality_score)

    @st.cache_data(show_spinner="🤖 Training Temporal Transformer (Attention Engine v8.0)...")
    def train_predict_transformer(df_ticker, lookback=90, forecast_days=30, sector_name=None, quality_score=50):
        """
        Drop-in replacement for train_predict_lstm using the pure Transformer architecture.
        Returns the same (path_array, return_pct, feature_importance) tuple.
        """
        import warnings
        warnings.filterwarnings('ignore')
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

            # ── Shared Feature Engineering (cached per ticker) ──
            feat = _precompute_features(df_ticker)
            if feat is None:
                return None, 0.0, {}
            data_scaled  = feat['data_scaled']
            price_scaler = feat['price_scaler']
            features     = feat['features']
            data         = feat['data']
            df           = feat['df']

            if len(data) < lookback + forecast_days:
                return None, 0.0, {}

            X, y = [], []
            for i in range(lookback, len(data_scaled) - forecast_days):
                X.append(data_scaled[i-lookback:i])
                y.append(data_scaled[i:i+forecast_days, 0])

            X = torch.FloatTensor(np.array(X)).to(device)
            y = torch.FloatTensor(np.array(y)).to(device)

            model = StockTransformer(input_size=len(features), d_model=64, nhead=4,
                                     num_layers=2, output_size=forecast_days).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = torch.nn.HuberLoss(delta=0.5)

            model.train()
            for epoch in range(60):
                indices = torch.randperm(X.size(0), device=device)
                for start_idx in range(0, X.size(0), 128):
                    idx = indices[start_idx:start_idx+128]
                    X_b, y_b = X[idx], y[idx]
                    
                    optimizer.zero_grad()
                    out = model(X_b)
                    
                    y_baseline = X_b[:, -1, 0].unsqueeze(1)
                    out = out + y_baseline
                    
                    loss = criterion(out, y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                import time; time.sleep(0.005)

            # Inference
            model.eval()
            with torch.no_grad():
                last_seq = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                y_baseline_inf = last_seq[:, -1, 0].unsqueeze(1)
                pred_scaled = (model(last_seq) + y_baseline_inf).cpu().numpy()[0]

            # Inverse-transform only price column
            price_scaler = MinMaxScaler(feature_range=(-1, 1))
            price_scaler.fit(data[:, 0:1])
            full_pred = np.zeros((forecast_days, len(features)))
            full_pred[:, 0] = pred_scaled
            forecast_raw = price_scaler.inverse_transform(full_pred[:, 0:1]).flatten()

            last_price = data[-1, 0]
            total_return = (forecast_raw[-1] / last_price - 1) if last_price > 0 else 0.0

            # ── v9.1: Real gradient attribution (not random) ──
            feat_imp = {}
            try:
                model.eval()
                last_seq_t = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device).requires_grad_(True)
                out_t = model(last_seq_t)
                torch.sum(out_t).backward()
                imp_t = torch.abs(last_seq_t.grad[0]).mean(dim=0).cpu().numpy()
                imp_t = imp_t / (imp_t.sum() + 1e-9) * 100
                feat_imp = {f: round(float(v), 1) for f, v in zip(features, imp_t)}
            except Exception:
                feat_imp = {f: round(100/len(features), 1) for f in features}

            return forecast_raw, total_return, feat_imp
        except Exception as e:
            return None, 0.0, {}

    @st.cache_data(show_spinner="🧬 Training PatchTST (SOTA Channel-Independent Engine v10.0)...")
    def train_predict_patchtst(df_ticker, lookback=120, forecast_days=30, sector_name=None, quality_score=50):
        """
        PatchTST Channel-Independent engine. Each of the 12 factors is processed
        by the SAME Transformer independently (no cross-channel noise), then averaged.
        Best for long-horizon, fundamental-driven forecasts.
        """
        import warnings
        warnings.filterwarnings('ignore')
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

            # ── Shared Feature Engineering (cached per ticker) ──
            feat = _precompute_features(df_ticker)
            if feat is None:
                return None, 0.0, {}
            data_scaled  = feat['data_scaled']
            price_scaler = feat['price_scaler']
            features     = feat['features']
            data         = feat['data']

            if len(data) < lookback + forecast_days:
                return None, 0.0, {}

            # ── Patch parameters ──
            patch_len = 16; stride = 8
            num_patches = (lookback - patch_len) // stride + 1

            X, y = [], []
            for i in range(lookback, len(data_scaled) - forecast_days):
                X.append(data_scaled[i-lookback:i])
                y.append(data_scaled[i:i+forecast_days, 0])
            X = torch.FloatTensor(np.array(X)).to(device)  # [N, T, C]
            y = torch.FloatTensor(np.array(y)).to(device)  # [N, forecast_days]

            model = StockPatchTST(
                c_in=len(features), context_window=lookback,
                target_window=forecast_days, patch_len=patch_len,
                stride=stride, d_model=64, nhead=4, num_layers=2
            ).to(device)
            optimizer  = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
            criterion  = torch.nn.HuberLoss(delta=0.5)
            scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

            model.train()
            for epoch in range(80):
                indices = torch.randperm(X.size(0), device=device)
                for start_idx in range(0, X.size(0), 128):
                    idx = indices[start_idx:start_idx+128]
                    X_b, y_b = X[idx], y[idx]
                    
                    optimizer.zero_grad()
                    out = model(X_b)
                    
                    y_baseline = X_b[:, -1, 0].unsqueeze(1)
                    out = out + y_baseline
                    
                    loss = criterion(out, y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                import time; time.sleep(0.005)

            # Inference
            model.eval()
            with torch.no_grad():
                last_seq_p   = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                y_baseline_p = last_seq_p[:, -1, 0].unsqueeze(1)
                pred_scaled  = (model(last_seq_p) + y_baseline_p).cpu().numpy()[0]

            # Inverse-transform price
            full_pred_p    = np.zeros((forecast_days, len(features)))
            full_pred_p[:, 0] = pred_scaled
            forecast_raw_p = price_scaler.inverse_transform(full_pred_p[:, 0:1]).flatten()

            last_price_p = data[-1, 0]
            total_return_p = (forecast_raw_p[-1] / last_price_p - 1) if last_price_p > 0 else 0.0

            # Gradient-based feature importance
            feat_imp_p = {}
            try:
                last_seq_grad = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device).requires_grad_(True)
                out_grad = model(last_seq_grad)
                torch.sum(out_grad).backward()
                imp_p = torch.abs(last_seq_grad.grad[0]).mean(dim=0).cpu().numpy()
                imp_p = imp_p / (imp_p.sum() + 1e-9) * 100
            except Exception:
                feat_imp_p = {f: round(100/len(features), 1) for f in features}

            return forecast_raw_p, total_return_p, feat_imp_p
        except Exception:
            return None, 0.0, {}

    @st.cache_data(show_spinner="Smart Blend: Training all 3 AI Engines (LSTM + Transformer + PatchTST)...")
    def train_predict_ensemble(df_ticker, lookback=90, forecast_days=30, sector_name=None, quality_score=50):
        """
        Performance-Weighted Ensemble: trains all 3 engines, evaluates each on the
        most recent holdout period (last forecast_days of known data), and blends
        their forecasts using weights proportional to 1/RMSE.
        """
        import warnings; warnings.filterwarnings('ignore')
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            feat = _precompute_features(df_ticker)
            if feat is None: return None, 0.0, {}, {}
            data_scaled, price_scaler, features, data, n_feat = feat['data_scaled'], feat['price_scaler'], feat['features'], feat['data'], feat['n_feat']
            if len(data) < lookback + 2 * forecast_days: return None, 0.0, {}, {}

            X_arr, y_arr = [], []
            for i in range(lookback, len(data_scaled) - forecast_days):
                X_arr.append(data_scaled[i-lookback:i]); y_arr.append(data_scaled[i:i+forecast_days, 0])
            X_t = torch.FloatTensor(np.array(X_arr)).to(device)
            y_t = torch.FloatTensor(np.array(y_arr)).to(device)
            if len(X_t) < 5: return None, 0.0, {}, {}

            def _eval_holdout(mdl):
                n_d = len(data_scaled)
                eval_x = data_scaled[n_d - lookback - forecast_days : n_d - forecast_days]
                actual_s = data_scaled[n_d - forecast_days : n_d, 0]
                mdl.eval()
                with torch.no_grad():
                    inp = torch.FloatTensor(eval_x).unsqueeze(0).to(device)
                    y_base_eval = inp[:, -1, 0].unsqueeze(1)  # Must match training residual logic
                    pred_s = (mdl(inp) + y_base_eval).cpu().numpy().flatten()[:forecast_days]
                fp = np.zeros((forecast_days, n_feat)); fp[:, 0] = pred_s
                fa = np.zeros((forecast_days, n_feat)); fa[:, 0] = actual_s
                pp = price_scaler.inverse_transform(fp[:, 0:1]).flatten()
                ap = price_scaler.inverse_transform(fa[:, 0:1]).flatten()
                # RMSE on price (absolute scale)
                rmse = float(np.sqrt(np.mean((pp - ap) ** 2)))
                # MAPE on daily returns (%) — meaningful regardless of price scale
                ap_ret = np.diff(ap) / (np.abs(ap[:-1]) + 1e-8)
                pp_ret = np.diff(pp) / (np.abs(pp[:-1]) + 1e-8)
                if len(ap_ret) > 0:
                    mape_ret = float(np.mean(np.abs(ap_ret - pp_ret)) * 100)
                else:
                    mape_ret = float(np.mean(np.abs((ap - pp) / (np.abs(ap) + 1e-8))) * 100)
                dir_acc = float(1.0 if (pp[-1] > pp[0]) == (ap[-1] > ap[0]) else 0.0)
                return rmse, mape_ret, dir_acc

            def _infer(mdl):
                mdl.eval()
                with torch.no_grad():
                    last_x = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                    y_base = last_x[:, -1, 0].unsqueeze(1)
                    pr = (mdl(last_x) + y_base).cpu().numpy().flatten()[:forecast_days]
                fp = np.zeros((forecast_days, n_feat)); fp[:, 0] = pr
                return price_scaler.inverse_transform(fp[:, 0:1]).flatten()

            results = {}
            import time as _time

            # (A) LSTM
            try:
                ml = StockLSTM(input_size=n_feat, hidden_size=64, num_layers=2, output_size=forecast_days).to(device)
                ol = torch.optim.Adam(ml.parameters(), lr=1e-3, weight_decay=1e-5); cl = torch.nn.HuberLoss(delta=1.0)
                ml.train()
                for _ in range(30):
                    idxs = torch.randperm(X_t.size(0), device=device)
                    for si in range(0, X_t.size(0), 128):
                        idx = idxs[si:si+128]; Xb, yb = X_t[idx], y_t[idx]; ybl = Xb[:, -1, 0].unsqueeze(1)
                        ol.zero_grad(); ls = cl(ml(Xb) + ybl, yb); ls.backward(); torch.nn.utils.clip_grad_norm_(ml.parameters(), 1.0); ol.step()
                    _time.sleep(0.005)
                path_l = _infer(ml); rm, mp, dr = _eval_holdout(ml)
                results['LSTM'] = {'path': path_l, 'rmse': rm, 'mape': mp, 'dir': dr, 'model': ml}
            except Exception: pass

            # (B) Transformer
            try:
                mt = StockTransformer(input_size=n_feat, d_model=64, nhead=4, num_layers=2, output_size=forecast_days).to(device)
                ot = torch.optim.Adam(mt.parameters(), lr=1e-3, weight_decay=1e-5); ct = torch.nn.HuberLoss(delta=0.5)
                mt.train()
                for _ in range(30):
                    idxs = torch.randperm(X_t.size(0), device=device)
                    for si in range(0, X_t.size(0), 128):
                        idx = idxs[si:si+128]; Xb, yb = X_t[idx], y_t[idx]; ybl = Xb[:, -1, 0].unsqueeze(1)
                        ot.zero_grad(); ls = ct(mt(Xb) + ybl, yb); ls.backward(); torch.nn.utils.clip_grad_norm_(mt.parameters(), 1.0); ot.step()
                    _time.sleep(0.005)
                path_t = _infer(mt); rm, mp, dr = _eval_holdout(mt)
                results['Transformer'] = {'path': path_t, 'rmse': rm, 'mape': mp, 'dir': dr, 'model': mt}
            except Exception: pass

            # (C) PatchTST
            try:
                patch_len, stride = 16, 8
                mp_m = StockPatchTST(c_in=n_feat, context_window=lookback, target_window=forecast_days, patch_len=patch_len, stride=stride, d_model=64, nhead=4, num_layers=2).to(device)
                op = torch.optim.Adam(mp_m.parameters(), lr=8e-4, weight_decay=1e-5); cp = torch.nn.HuberLoss(delta=0.5); sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=30, eta_min=1e-5)
                mp_m.train()
                for _ in range(30):
                    idxs = torch.randperm(X_t.size(0), device=device)
                    for si in range(0, X_t.size(0), 128):
                        idx = idxs[si:si+128]; Xb, yb = X_t[idx], y_t[idx]; ybl = Xb[:, -1, 0].unsqueeze(1)
                        op.zero_grad(); ls = cp(mp_m(Xb) + ybl, yb); ls.backward(); torch.nn.utils.clip_grad_norm_(mp_m.parameters(), 1.0); op.step()
                    sp.step(); _time.sleep(0.005)
                path_p = _infer(mp_m); rm, mp_v, dr = _eval_holdout(mp_m)
                results['PatchTST'] = {'path': path_p, 'rmse': rm, 'mape': mp_v, 'dir': dr, 'model': mp_m}
            except Exception: pass

            if not results: return None, 0.0, {}, {}
            
            # ── Regime-Aware Weighting with Minimum Threshold (v11.1) ──────────
            # Base: inverse-RMSE weighting
            best_rmse = min(v['rmse'] for v in results.values())
            inv_rmse = {k: 1.0 / max(v['rmse'], 0.01) for k, v in results.items()}
            
            # FIX: Filter out models with RMSE > 2x best model (too poor quality)
            inv_rmse = {k: v for k, v in inv_rmse.items() 
                       if results[k]['rmse'] <= 2.0 * best_rmse}
            
            if not inv_rmse:  # Fallback if all models filtered out
                inv_rmse = {k: 1.0 / max(v['rmse'], 0.01) for k, v in results.items()}
            
            total_inv = sum(inv_rmse.values())
            weights = {k: v / total_inv for k, v in inv_rmse.items()}
            
            # Regime boost: data-driven from model_performance_log analysis
            #   BULLISH/STRONG BULLISH → PatchTST wins 42.3% of time → +15% boost
            #   BEARISH/CAUTION       → Transformer & LSTM tied → +10% Transformer boost
            # Pre-refactor this read `'regime' in dir()` at module level, which was always False
            # (dir() inside a function never sees module globals) → the boost below never ran.
            # Kept disabled to preserve behaviour; the "data-driven" weights were never validated.
            _cur_regime = ""
            if "BULLISH" in _cur_regime and "PatchTST" in weights:
                weights["PatchTST"] *= 1.15
            elif "BEARISH" in _cur_regime or "CAUTION" in _cur_regime:
                if "Transformer" in weights: weights["Transformer"] *= 1.10
            
            # Renormalize
            _total_w = sum(weights.values())
            weights = {k: round(v / _total_w, 4) for k, v in weights.items()}
            blended = np.zeros(forecast_days)
            for k, v in results.items(): blended += weights[k] * v['path'][:forecast_days]
            last_price_e = data[-1, 0]; total_return_e = (blended[-1] / last_price_e - 1) if last_price_e > 0 else 0.0
            feat_imp_e = {f: 0.0 for f in features}
            try:
                last_seq_e = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                for k, v in results.items():
                    inp_g = last_seq_e.detach().clone().requires_grad_(True); torch.sum(v['model'](inp_g)).backward()
                    imp_g = torch.abs(inp_g.grad[0]).mean(dim=0).cpu().numpy(); imp_g = imp_g / (imp_g.sum() + 1e-9)
                    for i, f in enumerate(features): feat_imp_e[f] += weights[k] * float(imp_g[i]) * 100
                feat_imp_e = {f: round(v, 1) for f, v in feat_imp_e.items()}
            except Exception: feat_imp_e = {f: round(100/n_feat, 1) for f in features}
            metrics_dict = {k: {'RMSE': round(v['rmse'], 2), 'MAPE (%)': round(v['mape'], 2), 'Dir. Acc': f"{v['dir']*100:.0f}%", 'Weight': f"{weights[k]*100:.1f}%", 'Target': round(v['path'][-1], 2)} for k, v in results.items()}
            return blended, total_return_e, feat_imp_e, metrics_dict
        except Exception: return None, 0.0, {}, {}

    # ── PERSISTENT PERFORMANCE LOGGER ──
    import json as _json; from pathlib import Path as _Path; from datetime import datetime as _dt
    _LOG_PATH = _Path("model_performance_log.json")
    def _load_perf_log():
        if _LOG_PATH.exists():
            try: return _json.loads(_LOG_PATH.read_text())
            except: pass
        return {"logs": []}
    def _save_perf_log(d):
        try: _LOG_PATH.write_text(_json.dumps(d, indent=2))
        except: pass
    def _log_ensemble_run(ticker, horizon, vix_level, em):
        if not em: return
        anchor = max(em.keys(), key=lambda k: float(em[k]['Weight'].replace('%','')))
        entry = {"ts": _dt.now().isoformat()[:19], "ticker": ticker, "horizon": horizon, "vix": round(vix_level, 2), "regime": regime, "anchor": anchor,
                 "models": {k: {"rmse": v["RMSE"], "mape": v["MAPE (%)"], "weight": float(v["Weight"].replace("%",""))/100} for k, v in em.items()}}
        d = _load_perf_log(); d["logs"].append(entry); d["logs"] = d["logs"][-500:]; _save_perf_log(d)

    render_header("trending-up", "Price & Monte Carlo Forecasting", level="###")
    
    # ── AI STRATEGIST GUIDE (Synchronized with Master Tactical Regime) ──
    if regime == "STRONG BULLISH" or regime == "BULLISH":
        _rec_model, _rec_reason = "Neural v9.1 · Transformer (Direct 12F)", f"In a <b>{regime}</b> environment, prioritize <b>Pattern Recognition</b> & momentum capture via the Transformer ensemble."
    elif regime == "NEUTRAL / SIDEWAYS":
        _rec_model, _rec_reason = "Hybrid LSTM + Multi-Head Attention", f"Current <b>{regime}</b> regime favors <b>Temporal Stability</b>. LSTM is best for mean-reversion and sequential price discovery."
    else: # BEARISH / CAUTION
        _rec_model, _rec_reason = "PatchTST (Channel-Independent) + Monte Carlo", f"Tactical <b>{regime}</b> detected. Shift to <b>Structural Robustness</b>. PatchTST is less prone to noise during trend breakdowns."

    st.markdown(f"""
    <div style='background:rgba(52,152,219,0.08); border:1px solid #3498db; padding:16px; border-radius:8px; margin-bottom:25px;'>
        <div style='display:flex; align-items:center; margin-bottom:8px;'>{SVG_ICONS["brain"]} <b style='color:#3498db; font-size:1rem; margin-left:4px;'>ML Model Strategist</b> <span style='color:#8899aa; font-size:0.75rem; margin-left:8px;'>(rule of thumb — model choice by regime has not been validated)</span></div>
        <div style='font-size:0.92rem; color:#e0e0e0; line-height:1.5;'>Market Context: <b style='color:{regime_ui_color};'>{regime}</b><br>Suggested model: <b style='color:#00ffcc;'>{_rec_model}</b><br>Rationale: <i>{_rec_reason}</i></div>
        <div style='margin-top:12px; font-size:0.82rem; color:#8899aa; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px;'><b>Pro Tip:</b> LSTM+ARIMA anchors mean-reversion • Transformer excels in patterns • <b>PatchTST (v10.0)</b> gives high fundamental resolution — ideal for stable markets.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── ROW 1: Forecast Configuration (Horizontal Form) ──────────────────────
    with st.form("forecast_config_form"):
        fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1, 1, 1])
        with fcol1:
            # Sync active_ticker into fc_selector if available and not already set
            if 'fc_selector_form' not in st.session_state:
                _at = st.session_state.get('active_ticker', None)
                if _at and _at in current_universe:
                    st.session_state['fc_selector_form'] = _at
                
            fc_ticker = st.selectbox("Select Ticker to Forecast", current_universe, 
                                     format_func=format_ticker,
                                     index=None,
                                     placeholder="Choose a Ticker...",
                                     key="fc_selector_form")
        with fcol2:
            forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 7, key="fc_days_form")
        with fcol3:
            n_sims = st.selectbox("Monte Carlo Simulations", [500, 1000, 1500, 2000, 5000], index=3, key="n_sims_form")
        with fcol4:
            engine_mode = st.radio(
                "Core Engine",
                options=["LSTM Core", "Transformer", "PatchTST (SOTA)", "Smart Blend (Best of 3)"],
                index=3,
                key="engine_mode_form",
                help="LSTM Core: stable mean-reversion • Transformer: high-vol pattern recognition • PatchTST: channel-independent fundamentals • Smart Blend: trains all 3 engines and auto-weights them by accuracy (RMSE)."
            )
            
        run_forecast = st.form_submit_button("🎯 EXECUTE ML ENSEMBLE FORECAST", width="stretch", type="primary")

    # Initialize before the forecast block so the metrics panel never hits NameError
    ensemble_metrics = {}

    if run_forecast and fc_ticker:
        fc_ticker = st.session_state.fc_selector_form
        forecast_days = st.session_state.fc_days_form
        n_sims = st.session_state.n_sims_form
        engine_mode = st.session_state.engine_mode_form
        
        df_fc = prices_full[prices_full["ticker"] == fc_ticker].sort_values("date")
        ts = df_fc["price_close"].values
        
        # Pre-fetch Company data for Sector context
        co_data = companies_full[companies_full["ticker"] == fc_ticker].iloc[0] if not companies_full[companies_full["ticker"] == fc_ticker].empty else None
        sector_val = co_data['sector'] if co_data is not None else None
        company_val = co_data['company'] if co_data is not None and 'company' in co_data else fc_ticker
        
        # 1. ML Prediction — branch on engine_mode (Standardized lookback by horizon)
        if forecast_days <= 14:   std_lookback = 90
        elif forecast_days <= 45: std_lookback = 180
        else:                     std_lookback = 252

        drift_score    = compute_score(co_data) if co_data is not None else 50
        use_ensemble    = (engine_mode == "Smart Blend (Best of 3)")
        use_patchtst    = (engine_mode == "PatchTST (SOTA)")
        use_transformer = (engine_mode == "Transformer")
        if len(df_fc) < 30:
            st.warning(f"⚠️ Insufficient historical data ({len(df_fc)} days) to train the ML neural network. At least 30 days are required.")
            lstm_path, lstm_return, feat_imp = None, 0.0, {}
            st.session_state['ensemble_metrics'] = {}
        elif use_ensemble:
            with st.spinner(f"Smart Blend: Training all 3 ML engines ({std_lookback}D Lookback)..."):
                _ens = train_predict_ensemble(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            if _ens[0] is not None:
                lstm_path, lstm_return, feat_imp, _em = _ens
                st.session_state['ensemble_metrics'] = _em
                # Log this run for meta-evaluation
                _vix_now = float(prices_full[prices_full['ticker']=='^VIX']['price_close'].iloc[-1]) \
                    if not prices_full[prices_full['ticker']=='^VIX'].empty else 20.0
                _log_ensemble_run(fc_ticker, forecast_days, _vix_now, _em)
            else:
                st.warning("⚠️ Ensemble failed. Falling back to LSTM...")
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
                st.session_state['ensemble_metrics'] = {}
        elif use_patchtst:
            with st.spinner(f"🧬 Running PatchTST ({std_lookback}D Lookback)..."):
                lstm_path, lstm_return, feat_imp = train_predict_patchtst(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            if lstm_path is None:
                st.warning(f"⚠️ PatchTST needs {std_lookback}+ days. Falling back to LSTM...")
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            st.session_state['ensemble_metrics'] = {}
        elif use_transformer:
            with st.spinner(f"🤖 Running Transformer ({std_lookback}D Lookback)..."):
                lstm_path, lstm_return, feat_imp = train_predict_transformer(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            if lstm_path is None:
                st.warning(f"⚠️ Transformer needs {std_lookback}+ days. Falling back to LSTM...")
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            st.session_state['ensemble_metrics'] = {}
        else:  # LSTM Core
            with st.spinner(f"Running LSTM Core ({std_lookback}D Lookback)..."):
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            st.session_state['ensemble_metrics'] = {}
        
        # 2. News Sentiment (High-Accuracy FinBERT) using Google News
        import feedparser
        import urllib.parse
        _q_fc = urllib.parse.quote(f"{company_val} stock when:7d") if company_val else urllib.parse.quote(f"{fc_ticker} stock when:7d")
        rss_url = f"https://news.google.com/rss/search?q={_q_fc}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        titles = [entry.get("title", "").split(" - ")[0] for entry in feed.entries[:10]]
        
        # FIX: Add warning when no news found
        if not titles:
            st.warning(f"⚠️ No recent news found for {company_val}. Sentiment defaulted to neutral (0.0).")
            avg_sent = 0
        else:
            avg_sent = analyze_sentiment_finbert(titles)
        
        # 4. Monte Carlo Simulation (AI-Enhanced & Dynamic Volatility)
        returns = df_fc["daily_return_pct"].dropna() / 100
        mu = returns.mean()
        sigma_long_term = returns.std()
        
        # Calculate current 'heat' (14-day rolling volatility)
        sigma_current = returns.tail(14).std() if len(returns) >= 14 else sigma_long_term
        
        last_price = ts[-1]
        
        drift_bias = 0
        if drift_score >= 75: drift_bias += 0.0005 
        elif drift_score <= 40: drift_bias -= 0.0005 
        drift_bias += (avg_sent * 0.001) 
        if lstm_return is not None and lstm_return > 0.05: drift_bias += 0.0005
        
        # ── Phase 7: Monte Carlo GARCH(1,1) (Volatility Clustering) ───────────
        try:
            # FIX: Use rescale=True to let arch_model handle scaling automatically
            # This eliminates manual scaling bugs and improves numerical stability
            am = arch_model(returns.tail(500), vol='Garch', p=1, q=1, dist='Normal', rescale=True)
            res = am.fit(disp='off')
            
            # Forecast volatility term structure for the horizon
            forecasts = res.forecast(horizon=forecast_days)
            # Variance -> Std Dev (rescale=True handles units automatically)
            sigma_forecast = np.sqrt(forecasts.variance.values[-1, :])
            
            # Ensure no zero/nan vol (fallback to long-term avg)
            sigma_forecast = np.nan_to_num(sigma_forecast, nan=sigma_long_term)
            sigma_forecast[sigma_forecast == 0] = sigma_long_term
            
        except Exception:
            # Robust Fallback to Mean Reversion (OU Process) if GARCH fails to converge
            kappa = 0.1 
            sigma_forecast = []
            s_t = sigma_current
            for _ in range(forecast_days):
                s_t = s_t + kappa * (sigma_long_term - s_t)
                sigma_forecast.append(s_t)
            
        # ── Phase 7.5: Monte Carlo — AI-Anchored GBM ────────────────────────────
        # Best Practice: use AI ensemble's implied drift + residual-calibrated vol
        # instead of raw historical mean return (which ignores the AI's forward view).

        # (A) AI-IMPLIED DRIFT: annualized daily drift from the AI forecast path
        if lstm_path is not None and len(lstm_path) >= 2 and last_price > 0:
            # Log-return implied by AI path from today to horizon end
            ai_total_log_return = np.log(lstm_path[-1] / last_price)
            mu_ai = ai_total_log_return / forecast_days   # per-day log drift
        else:
            mu_ai = mu + drift_bias  # fallback to historical if AI path unavailable

        # (B) RESIDUAL-CALIBRATED VOLATILITY:
        # Measure how much actual recent prices deviated from the AI's in-sample fit.
        # We approximate this by: residual_vol = std of (actual_return - AI implied step)
        # If unavailable, blend GARCH vol with rolling 21-day realized vol.
        try:
            actual_recent = df_fc['price_close'].values[-forecast_days-1:]
            if lstm_path is not None and len(actual_recent) >= 2:
                ai_step_returns = np.diff(np.log(lstm_path + 1e-9))[:len(actual_recent)-1]
                actual_step_returns = np.diff(np.log(actual_recent + 1e-9))
                min_len = min(len(ai_step_returns), len(actual_step_returns))
                residuals = actual_step_returns[:min_len] - ai_step_returns[:min_len]
                residual_vol = float(np.std(residuals)) if min_len > 2 else sigma_current
            else:
                residual_vol = sigma_current
            # Blend: 60% GARCH structure + 40% AI residual (retains clustering + calibration)
            sigma_blended = np.array([
                0.6 * float(s) + 0.4 * residual_vol for s in sigma_forecast
            ])
            sigma_blended = np.clip(sigma_blended, sigma_long_term * 0.3, sigma_long_term * 4.0)
        except Exception:
            sigma_blended = np.array(sigma_forecast)

        # (C) SIMULATE PATHS anchored on AI-implied drift, noise from residual vol
        # (C) SIMULATE PATHS (Vectorized NumPy implementation for M3 Speed)
        Z = np.random.normal(size=(forecast_days, n_sims))
        s_v = sigma_blended.reshape(-1, 1) # (days, 1) for broadcasting
        
        # Calculate all log-returns in one shot (GBM formula: r = (mu - 0.5*sigma^2) + sigma*Z)
        daily_log_rets = (mu_ai - 0.5 * s_v**2) + (s_v * Z)
        
        # Prepend zeros row for starting point (Price at T=0)
        cum_log_rets = np.vstack([np.zeros(n_sims), np.cumsum(daily_log_rets, axis=0)])
        
        # Final price paths: P_t = P_0 * exp(sum of daily log rets)
        simulated_paths = last_price * np.exp(cum_log_rets)

        
        # 1.5 Backtest Accuracy (Diagnostic) — Dynamic Horizon Sync (Phase 8)
        with st.spinner(f"Validating {forecast_days}-Day Accuracy..."):
            precision_score, mape_raw, naive_mape = calculate_backtest_accuracy(df_fc, sector_name=sector_val, quality_score=drift_score, test_size=forecast_days)

        # ── ROW 2: AI Metrics (Horizontal Cards) ─────────────────────────────
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("ML Ensemble Target", f"€{lstm_path[-1]:.2f}" if lstm_path is not None else "N/A", delta=f"{lstm_return*100:.2f}%" if lstm_return else "N/A")
            if lstm_path is not None:
                st.session_state[f"ai_target_for_de_{fc_ticker}"] = float(lstm_path[-1])
                st.caption("→ Shown as TP1 in the Stock Analysis signal matrix (experimental; not used by the Decision)")
        
        with mcol2:
            sent_label = "Bullish" if avg_sent > 0.1 else "Bearish" if avg_sent < -0.1 else "Neutral"
            st.metric("News Sentiment Mood", sent_label, delta=f"{avg_sent:.2f}")
        with mcol3:
            # Smart Money Spirit (Unified v6.0 with sector awareness)
            sm_result = get_sm_spirit_unified_v2(df_fc, sector=str(sector_val) if sector_val else "Unknown")
            sm_signal = sm_result["signal"]
            sm_strength = sm_result["strength"]
            sm_layer = sm_result["layer"]
            vol_quality = sm_result.get("volume_quality", 0)
            mfi_confirm = sm_result.get("mfi_confirm", False)
            
            # Enhanced display with volume quality and MFI confirmation
            sm_display = f"{sm_signal} ({sm_strength}/100)"
            if mfi_confirm:
                sm_display += " ✓MFI"
            
            st.metric(
                "Smart Money Spirit", 
                sm_display, 
                delta=f"{sm_layer} · Vol Q: {vol_quality}/100" if sm_layer != "NONE" else "No Signal"
            )
        with mcol4:
            p_label = f"Holdout Error vs Naive ({forecast_days}d)"
            if mape_raw is not None and naive_mape:
                _skill = 1 - mape_raw / naive_mape   # >0 = better than "no change"
                st.metric(p_label, f"{mape_raw*100:.1f}% vs {naive_mape*100:.1f}%",
                          delta=f"{_skill:+.0%} skill vs no-change forecast",
                          delta_color="normal" if _skill > 0 else "inverse",
                          help="Mean absolute % error of the model on the last unseen window, next to a "
                               "forecast that simply assumes today's price. One window is a weak test; "
                               "do not act on the forecast unless skill is positive consistently.")
            else:
                st.metric(p_label, "N/A")
            
        # Highlight divergence
        if (sent_label == "Bearish" and sm_signal == "ACCUMULATION") or (sent_label == "Bullish" and sm_signal == "DISTRIBUTION"):
            div_type = "BULLISH DIVERGENCE (Smart Money Accumulating despite Retail Fear)" if sent_label == "Bearish" else "BEARISH DIVERGENCE (Smart Money Distributing despite Retail Greed)"
            div_color = "#2ecc71" if sent_label == "Bearish" else "#e74c3c"
            div_icon = "📈" if sent_label == "Bearish" else "📉"
            st.markdown(f"<div style='margin-top:10px; padding:12px 18px; background:linear-gradient(90deg, {div_color}22, rgba(0,0,0,0)); border-left:4px solid {div_color}; border-radius:6px;'><b style='color:{div_color}; font-size:1.0rem;'>{div_icon} Divergence (unvalidated): {div_type}</b><br><span style='font-size:0.85rem; color:#ccc;'>Institutions and Smart Money are actively positioning in direct opposition to retail sentiment (Strength: {sm_strength}/100, {sm_layer} Layer). This severe dislocation heavily tilts risk/reward for a contrarian entry. <b>Not a validated edge — treat as a prompt for further research.</b></span></div>", unsafe_allow_html=True)

        # ── AI TRADING SIGNATURE ─────────────────────────────────────────────
        # Pre-compute all levels for the card
        p5_final   = np.percentile(simulated_paths[-1, :], 5)
        p10_final  = np.percentile(simulated_paths[-1, :], 10)
        p90_final  = np.percentile(simulated_paths[-1, :], 90)
        p95_final  = np.percentile(simulated_paths[-1, :], 95)
        _ai_target = float(lstm_path[-1]) if lstm_path is not None else last_price
        _ai_stop   = float(p10_final)
        _ai_tp2    = float(p90_final)
        _ai_upside = (lstm_return * 100) if lstm_return is not None else 0

        # ── XGBoost Directional Signal (4th Pillar) ─────────────────────────
        with st.spinner("🌲 XGBoost Classifier running..."):
            xgb_signal, xgb_conf, xgb_imp = run_xgboost_signal(df_fc, horizon_days=min(5, forecast_days))

        # ── Conviction Score (4-Pillar: 0-4) ────────────────────────────────
        _conv_pts  = 0
        _conv_pts += 1 if _ai_upside >= 3 else 0
        _conv_pts += 1 if sm_signal == "ACCUMULATION" else 0
        _conv_pts += 1 if avg_sent > 0.05 else 0
        _conv_pts += 1 if xgb_signal == "BUY" else 0

        # R/R based on Monte Carlo bands
        _sig_risk   = last_price - _ai_stop
        _sig_reward = _ai_target - last_price
        _sig_rr     = (_sig_reward / _sig_risk) if _sig_risk > 0 else 0

        # ── Executive Verdict ────────────────────────────────────────────────
        if _conv_pts >= 4 and _sig_rr >= 1.5:
            _sig_verdict, _sig_color, _sig_badge = "STRONG LONG", "#00ffcc", "HIGH CONVICTION"
            _sig_desc = (f"All 4 pillars aligned: +{_ai_upside:.1f}% upside, Smart Money Accumulation, "
                         f"{sent_label} sentiment, and XGBoost signals BUY ({xgb_conf*100:.0f}% confidence). "
                         f"A {_sig_rr:.1f}x R/R setup — ideal for a full position.")
        elif _conv_pts == 3 and _sig_rr >= 1.5:
            _sig_verdict, _sig_color, _sig_badge = "STRONG LONG", "#00ffcc", "HIGH CONVICTION"
            _sig_desc = (f"3 of 4 pillars aligned: Projects +{_ai_upside:.1f}% upside, "
                         f"institutions in Accumulation, sentiment {sent_label}. "
                         f"XGBoost: {xgb_signal} ({xgb_conf*100:.0f}%). R/R {_sig_rr:.1f}x.")
        elif _conv_pts >= 2 and _sig_rr >= 1.0:
            _sig_verdict, _sig_color, _sig_badge = "BUY / ACCUMULATE", "#2ecc71", "MODERATE CONVICTION"
            _sig_desc = (f"2+ pillars constructive. Target €{_ai_target:.2f} ({_ai_upside:+.1f}%), "
                         f"Smart Money: {sm_signal}, XGBoost: {xgb_signal} ({xgb_conf*100:.0f}%). "
                         f"R/R {_sig_rr:.1f}x — partial position entry supported.")
        elif _ai_upside <= -3:
            _sig_verdict, _sig_color, _sig_badge = "REDUCE / HEDGE", "#e74c3c", "BEARISH SIGNAL"
            _sig_desc = (f"Model projects {_ai_upside:.1f}% downside to €{_ai_target:.2f}. "
                         f"Smart Money: {sm_signal}, XGBoost: {xgb_signal}. "
                         f"Reduce exposure or hedge until price stabilizes above €{_ai_stop:.2f}.")
        elif _conv_pts == 0:
            _sig_verdict, _sig_color, _sig_badge = "AVOID / WAIT", "#e74c3c", "NO CONVICTION"
            _sig_desc = (f"All 4 pillars negative: Upside {_ai_upside:+.1f}%, Smart Money {sm_signal}, "
                         f"sentiment {sent_label}, XGBoost {xgb_signal}. Stay flat.")
        else:
            _sig_verdict, _sig_color, _sig_badge = "NEUTRAL / MONITOR", "#f1c40f", "MIXED SIGNALS"
            _sig_desc = (f"Conflicting signals: Projects {_ai_upside:+.1f}% to €{_ai_target:.2f}. "
                         f"XGBoost: {xgb_signal} ({xgb_conf*100:.0f}%), Smart Money: {sm_signal}. "
                         f"Monitor for confluence before entry.")

        # ── Reasoning pills ─────────────────────────────────────────────────
        def _pill(label, value, ok):
            c = "#2ecc71" if ok else "#e74c3c"
            return (f"<span style='display:inline-flex; align-items:center; gap:5px; background:rgba(255,255,255,0.05); "
                    f"border:1px solid {c}55; border-radius:20px; padding:4px 10px; font-size:0.78rem; margin:3px;'>"
                    f"<span style='color:{c}; font-weight:700;'>{'✓' if ok else '✗'}</span> "
                    f"<span style='color:#ccc;'>{label}:</span> "
                    f"<span style='color:#fff; font-weight:700;'>{value}</span></span>")

        _pill_ai   = _pill("Upside",    f"{_ai_upside:+.1f}%",  _ai_upside >= 3)
        _pill_sm   = _pill("Smart Money",  f"{sm_signal} ({sm_strength})",  sm_signal == "ACCUMULATION")
        _pill_sent = _pill("Sentiment",    sent_label,             avg_sent > 0.05)
        _pill_rr   = _pill("R/R",          f"{_sig_rr:.1f}x",      _sig_rr >= 1.5)
        _ml_skill = (1 - mape_raw / naive_mape) if (mape_raw is not None and naive_mape) else None
        _pill_prec = _pill("ML vs naive", f"{_ml_skill:+.0%}" if _ml_skill is not None else "N/A", (_ml_skill or 0) > 0)
        _xgb_label = f"XGB {xgb_signal} ({xgb_conf*100:.0f}%)"
        _pill_xgb  = _pill("XGBoost Signal", _xgb_label, xgb_signal == "BUY")

        _unc_str = f"±{mape_raw*100:.1f}% CI" if mape_raw else ""
        _vix_now_sig = float(prices_full[prices_full['ticker']=='^VIX']['price_close'].iloc[-1]) \
            if not prices_full[prices_full['ticker']=='^VIX'].empty else 20.0
        _playbook = ("Mean Reversion / Range Trading" if _vix_now_sig > 25
                     else "Trend Following / Breakout" if _vix_now_sig < 15
                     else "Selective / Stock Picker's Market")

        def _hex_rgb(h): h=h.lstrip('#'); return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
        _bg_rgb = _hex_rgb(_sig_color)

        html_content = f"""
<div style='background:rgba(10,15,25,0.7); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:22px 26px; margin:18px 0;'>
<!-- Header Row -->
<div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; margin-bottom:18px;'>
<div>
<div style='font-size:0.65rem; color:#8899aa; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;'>AI Trading Signature</div>
<div style='font-size:1.8rem; font-weight:900; color:{_sig_color}; text-shadow:0 0 20px rgba({_bg_rgb},0.5); line-height:1;'>{_sig_verdict}</div>
<div style='font-size:0.75rem; color:{_sig_color}; background:rgba({_bg_rgb},0.12); border:1px solid rgba({_bg_rgb},0.35); border-radius:20px; display:inline-block; padding:2px 10px; margin-top:6px;'>{_sig_badge}</div>
</div>
<div style='text-align:right;'>
<div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; margin-bottom:4px;'>VIX Context</div>
<div style='font-size:1.1rem; font-weight:700; color:#f1c40f;'>VIX {_vix_now_sig:.1f}</div>
<div style='font-size:0.78rem; color:#aaa;'>{_playbook}</div>
</div>
</div>
<!-- Signal Pills -->
<div style='margin-bottom:16px;'>{_pill_ai}{_pill_sm}{_pill_sent}{_pill_rr}{_pill_prec}{_pill_xgb}</div>
<!-- Rationale -->
<div style='font-size:0.88rem; color:#dde; line-height:1.6; margin-bottom:18px; border-left:3px solid rgba({_bg_rgb},0.6); padding-left:14px;'>{_sig_desc}</div>
<!-- Trade Setup Snapshot -->
<div style='border-top:1px solid rgba(255,255,255,0.08); padding-top:16px;'>
<div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Trade Setup Snapshot</div>
<div style='display:grid; grid-template-columns:repeat(5,1fr); gap:8px; font-size:0.82rem;'>
<div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 12px; border-top:2px solid #3498db;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>CURRENT PRICE</div>
<div style='color:#fff; font-weight:800; font-size:1.05rem;'>€{last_price:.2f}</div>
</div>
<div style='background:rgba(46,204,113,0.08); border-radius:8px; padding:10px 12px; border-top:2px solid #2ecc71;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>ENTRY (NOW)</div>
<div style='color:#2ecc71; font-weight:800; font-size:1.05rem;'>€{last_price:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>{forecast_days}d forecast</div>
</div>
<div style='background:rgba(231,76,60,0.08); border-radius:8px; padding:10px 12px; border-top:2px solid #e74c3c;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>STOP (MC P10)</div>
<div style='color:#e74c3c; font-weight:800; font-size:1.05rem;'>€{_ai_stop:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>Risk: {((last_price-_ai_stop)/last_price*100):.1f}%</div>
</div>
<div style='background:rgba(0,255,204,0.06); border-radius:8px; padding:10px 12px; border-top:2px solid #00ffcc;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>TARGET 1 (ML)</div>
<div style='color:#00ffcc; font-weight:800; font-size:1.05rem;'>€{_ai_target:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>{_unc_str} · {_ai_upside:+.1f}%</div>
</div>
<div style='background:rgba(52,152,219,0.06); border-radius:8px; padding:10px 12px; border-top:2px solid #3498db;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>TARGET 2 (MC P90)</div>
<div style='color:#3498db; font-weight:800; font-size:1.05rem;'>€{_ai_tp2:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>Extended scenario</div>
</div>
</div>
<!-- R/R Progress Bar -->
<div style='margin-top:14px; display:flex; align-items:center; gap:12px;'>
<div style='color:#8899aa; font-size:0.75rem; white-space:nowrap;'>R/R Ratio</div>
<div style='flex:1; background:rgba(255,255,255,0.08); border-radius:4px; height:8px; position:relative; overflow:hidden;'>
<div style='width:{min(100, _sig_rr/3.0*100):.0f}%; height:100%; background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71,#00ffcc); border-radius:4px;'></div>
</div>
<div style='color:{_sig_color}; font-weight:800; font-size:0.9rem; white-space:nowrap;'>{_sig_rr:.2f}x</div>
<div style='color:#8899aa; font-size:0.75rem; white-space:nowrap;'>{'FAVORABLE' if _sig_rr>=1.5 else 'MARGINAL' if _sig_rr>=1.0 else 'POOR'}</div>
</div>
<!-- 90% Confidence Interval note -->
<div style='margin-top:10px; font-size:0.78rem; color:#8899aa; text-align:center;'>
Monte Carlo 90% CI: <b style='color:#fff;'>€{p5_final:.2f}</b> ↔ <b style='color:#fff;'>€{p95_final:.2f}</b> &nbsp;·&nbsp; {forecast_days}-Day Horizon &nbsp;·&nbsp; {n_sims:,} simulations
</div>
</div>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)

        # ── ROW 3: Main Chart (Full Width) ── (Moved to Top) ───────────────────
        render_header("ai", f"AI Ensemble vs Stochastic Monte Carlo: {fc_ticker}")
        fig_fc = go.Figure()
        # Include today's date so all lines start from the last known price point
        future_dates = pd.date_range(start=df_fc["date"].max(), periods=forecast_days+1, freq='B')
        
        for i in range(min(n_sims, 50)): 
            fig_fc.add_trace(go.Scatter(x=future_dates, y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
        
        mean_path = simulated_paths.mean(axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_path, name="Monte Carlo Mean Path", line=dict(color="rgba(241, 196, 15, 0.5)", width=2, dash="dash")))
        
        if lstm_path is not None:
            # Prepend today's price to visually close the gap on the chart
            lstm_plot_y = np.insert(lstm_path, 0, last_price)
            
            # ── Ensemble Uncertainty Bands (Calibrated from Backtest MAPE) ─────────
            if mape_raw is not None:
                # Temporal Confidence Decay: band widens with sqrt(t)
                time_decay = np.zeros(len(lstm_plot_y))
                time_decay[1:] = np.sqrt(np.arange(1, len(lstm_path) + 1) / len(lstm_path))
                lstm_upper = lstm_plot_y * (1 + mape_raw * time_decay)
                lstm_lower = lstm_plot_y * (1 - mape_raw * time_decay)
                # Shaded confidence region
                fig_fc.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(lstm_upper) + list(lstm_lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(0,229,255,0.08)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'Ensemble ±{mape_raw*100:.1f}% Confidence',
                    showlegend=True
                ))
            # Central Ensemble path (on top)
            fig_fc.add_trace(go.Scatter(
                x=future_dates, y=lstm_plot_y,
                name="AI Ensemble Most Likely Path",
                line=dict(color="#00E5FF", width=4)
            ))
        
        p10 = np.percentile(simulated_paths, 10, axis=1)
        p90 = np.percentile(simulated_paths, 90, axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p10, name="Lower Risk Bound (90%)", line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dot")))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p90, name="Upper Reward Bound (90%)", line=dict(color="rgba(0,255,0,0.5)", width=2, dash="dot")))
 
        fig_fc.update_layout(template="plotly_dark", height=600, yaxis_title="Price (€)", margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("---")

        # ── ROW 2.5: Intelligence Diagnostic (Breakdown) ──────────────────────
        _fc_meta = companies_full[companies_full['ticker'] == fc_ticker]
        render_header("activity", "AI Reasoning & Diagnostic Insight")
        dcol1, dcol2 = st.columns([1, 1])
        
        with dcol1:
            # Model Input Reasoning (SHAP)
            render_header("activity", "Model Input Reasoning (SHAP)")
            if feat_imp:
                pretty_feat_map = {
                    'price_close': 'Price Level',
                    'daily_return_pct': 'Volatility/Return',
                    'spy_ret': 'Market (SPY)', 
                    'vix_ret': 'Fear Index (VIX)',
                    'vol_surge': 'Volume Spike', 
                    'quality_score_norm': 'Quality Score'
                }
                imp_df = pd.DataFrame([
                    {'Feature': pretty_feat_map.get(k, k), 'Weight (%)': v}
                    for k, v in feat_imp.items()
                ]).sort_values('Weight (%)', ascending=True)

                fig_imp = px.bar(
                    imp_df, x='Weight (%)', y='Feature', orientation='h',
                    template="plotly_dark", height=300,
                    color='Weight (%)', color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(xaxis_title="Influence (%)", showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Insufficient data for SHAP analysis.")
            
        with dcol2:
            # ── Weighted Ensemble Metrics Panel (persisted via session_state) ──
            _em_display = st.session_state.get('ensemble_metrics', {})
            if _em_display:
                st.markdown("")
                render_header("activity", "Ensemble Performance Breakdown")
                rows_html = ""
                for model_name, m in _em_display.items():
                    icon_key = "brain" if model_name == "LSTM" else "bot" if model_name == "Transformer" else "dna"
                    icon_svg = SVG_ICONS[icon_key].replace('width="18"','width="14"').replace('height="18"','height="14"')
                    w_pct = float(m["Weight"].replace("%", ""))
                    bar_color = "#00ffcc" if w_pct == max(float(v["Weight"].replace("%", "")) for v in _em_display.values()) else "#3498db"
                    conf_score = max(0.0, 100.0 - m["MAPE (%)"])
                    conf_color = "#2ecc71" if conf_score >= 90 else "#f1c40f" if conf_score >= 80 else "#e74c3c"
                    rows_html += f"""
                    <tr>
                        <td style='padding:8px 12px; font-weight:600;'>{icon_svg} {model_name}</td>
                        <td style='padding:8px 12px; text-align:center; color:#e74c3c;'>€{m["RMSE"]}</td>
                        <td style='padding:8px 12px; text-align:center; color:#e67e22;'>{m["MAPE (%)"]:.1f}%</td>
                        <td style='padding:8px 12px; text-align:center; font-weight:700; color:{conf_color};'>{conf_score:.1f}%</td>
                        <td style='padding:8px 12px; text-align:center;' title='0% happens when mean-reverting models predict flatlines during a trending test set.'>{m["Dir. Acc"]}</td>
                        <td style='padding:8px 12px; text-align:center; font-weight:700; color:#f1c40f;'>€{m.get('Target', 0):.2f}</td>
                        <td style='padding:8px 12px; min-width:120px;'>
                            <div style='display:flex; align-items:center; gap:6px;'>
                                <div style='background:{bar_color}; height:8px; border-radius:4px; width:{w_pct:.0f}%; max-width:80px;'></div>
                                <span style='color:{bar_color}; font-weight:700; font-size:0.9rem;'>{m["Weight"]}</span>
                            </div>
                        </td>
                    </tr>"""
                st.markdown(f"""
                <table style='width:100%; border-collapse:collapse; font-size:0.88rem; color:#e0e0e0;'>
                    <thead>
                        <tr style='border-bottom:1px solid rgba(255,255,255,0.15); color:#8899aa; font-size:0.78rem; text-transform:uppercase;'>
                            <th style='padding:6px 12px; text-align:left;'>Model</th>
                            <th style='padding:6px 12px; text-align:center;'>RMSE (€)</th>
                            <th style='padding:6px 12px; text-align:center;'>MAPE</th>
                            <th style='padding:6px 12px; text-align:center;' title='Confidence Score (100 - MAPE)'>Confidence <span style='cursor:help;'>ⓘ</span></th>
                            <th style='padding:6px 12px; text-align:center;' title='Directional Accuracy evaluated on the holdout window'>Dir. Acc <span style='cursor:help;'>ⓘ</span></th>
                            <th style='padding:6px 12px; text-align:center;'>Target Vote</th>
                            <th style='padding:6px 12px; text-align:left;'>Weight</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)
                st.caption("💡 Weight ∝ 1/RMSE — the model with the lowest error has the highest influence on the final forecast.")
                st.markdown("<div style='font-size:0.85rem; color:#8899aa; margin-top:4px;'><b>Note on Dir. ACC 0%:</b> LSTM & Transformer are mathematically prone to 0% Directional Accuracy because they tend to output mean-reverting flatlines. If the real price trends slightly, the strict binary direction check fails. <b>PatchTST</b>, functioning as a structural forecaster, is more likely to yield 100% on trajectory direction.</div>", unsafe_allow_html=True)

            # ── Meta Intelligence Panel ──────────────────────────────────────
            with st.expander("🧪 Meta Intelligence — Anchor History & VIX Regime Analysis", expanded=False):
                _perf_data = _load_perf_log()["logs"]
                if len(_perf_data) < 2:
                    st.info("📊 Insufficient history. Please run Smart Blend at least twice to build Regime analysis.")
                else:
                    # Section A: Recent Anchor History
                    st.markdown("​**Anchor Model by Run (Latest 20):**")
                    anchor_rows = [{
                        "Time": e["ts"], "Ticker": e["ticker"],
                        "Horizon": f"{e['horizon']}D",
                        "VIX": e["vix"],
                        "Regime": "⬆️ High" if e["regime"]=="high_vix" else "⬇️ Low",
                        "Anchor Model": e["anchor"]
                    } for e in _perf_data[-20:][::-1]]
                    st.dataframe(pd.DataFrame(anchor_rows), width="stretch", hide_index=True)

                    # Section B: VIX-Regime Performance Chart
                    if len(_perf_data) >= 3:
                        regime_rows = []
                        for e in _perf_data:
                            for mn, mv in e["models"].items():
                                regime_rows.append({
                                    "Model": mn,
                                    "Regime": "High VIX (>25)" if e["regime"]=="high_vix" else "Low VIX (≤25)",
                                    "RMSE": mv["rmse"]
                                })
                        regime_df = pd.DataFrame(regime_rows)
                        avg_r = regime_df.groupby(["Model","Regime"])["RMSE"].mean().reset_index()
                        if not avg_r.empty:
                            st.markdown("​**📈 Model RMSE by VIX Regime:**")
                            fig_meta = px.bar(
                                avg_r, x="Model", y="RMSE", color="Regime",
                                barmode="group", template="plotly_dark", height=260,
                                color_discrete_map={"High VIX (>25)": "#e74c3c", "Low VIX (≤25)": "#2ecc71"},
                                labels={"RMSE": "Avg RMSE (€)"}
                            )
                            fig_meta.update_layout(margin=dict(t=10,b=0,l=0,r=0),
                                                   legend=dict(orientation="h", y=1.12))
                            st.plotly_chart(fig_meta, use_container_width=True)
                            st.caption("💡 Lower bar = more effective model in that market regime. Key question: Does LSTM or Transformer perform better during VIX spikes?")

            st.markdown("<br>", unsafe_allow_html=True)

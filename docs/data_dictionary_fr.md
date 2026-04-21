# Dictionnaire des Données et Lignage (Data Lineage)

Cette documentation clarifie l'origine de tous les points de données dans l'entrepôt `stock_dw.duckdb`. Elle précise quels champs proviennent directement du fournisseur amont (Yahoo Finance) et quels champs sont calculés localement via le pipeline ETL.

## 1. Couche Brute (Schéma `raw`)

Toutes les tables du schéma `raw` sont issues à 100% de **Yahoo Finance (via l'API `yfinance`)**. Aucun calcul financier ni transformation n'intervient à ce niveau; seules la validation des types et la standardisation sont effectuées.

| Table | Source | Description |
| :--- | :--- | :--- |
| `raw.company_info` | Yahoo Finance (`.info`) | Métadonnées de base, ratios financiers statiques (P/E, Dette, Marges) et classification sectorielle (Secteur, Industrie). Remarque : Le secteur est remplacé et priorisé par le fichier local `config/tickers.yaml`. |
| `raw.stock_prices` | Yahoo Finance (`.history`) | Prix quotidiens EOD (Ouverture, Haut, Bas, Clôture, Volume). |
| `raw.quarterly_financials` | Yahoo Finance | Sélection des données des états financiers trimestriels (Facturation, Résultat Net, EPS, Capitaux propres). |
| `raw.historical_financials` | Yahoo Finance | Sélection des données des états financiers annuels. |
| `raw.cashflows` | Yahoo Finance (`.cashflow`) | Éléments de flux de trésorerie sur les douze derniers mois (TTM : Rachats d'actions et dividendes payés). |

---

## 2. Couche de Présentation (Schéma `marts`)

Le schéma `marts` combine les données brutes avec des calculs locaux approfondis (Indicateurs Techniques, Momentum, Taux de croissance) via des requêtes SQL dans `etl/transform.py`.

### A. `marts.dim_companies`

Cette table sert de pivot pour les filtres de notation sur l'interface (Screener).

> [!NOTE] 
> La majorité des métriques telles que `forward_pe`, `roe` ou `debt_to_equity` sont lues directement depuis la table brute. Les champs listés ci-dessous sont **calculés localement**.

| Colonne | Source | Calcul / Formule |
| :--- | :--- | :--- |
| `cap_category` | Calculé | Classé par taille de marché : <br> `≥ $1T` = Mega-Cap <br> `$200B-$1T` = Large-Cap <br> `$10B-$200B` = Mid-Cap <br> `< $10B` = Small-Cap |
| `buyback_yield_pct` | Calculé | `(buyback_ttm / market_cap) * 100` |
| `net_payout_yield_pct` | Calculé | Rendement cumulé des rachats d'actions et la distribution de dividendes. |
| `fcf_margin` | Calculé | `(free_cashflow / revenue_ttm) * 100` |
| `fmi_rev_acceleration` | Calculé | Différence entre la croissance QoQ des revenus T(actuel) et T-1(précédent). |
| `fmi_eps_acceleration` | Calculé | Différence entre la croissance QoQ du BPA (EPS) T(actuel) et T-1(précédent). |
| `fmi_margin_trend` | Calculé | Croissance YoY du BPA - Croissance YoY des revenus (évalue le levier d'exploitation opérationnel). |
| `fmi_quarters_of_growth` | Calculé | Nombre consécutif des derniers trimestres avec croissance positive du revenu et de l'EPS (YoY). |

### B. `marts.fct_daily_returns`

| Colonne | Source | Calcul / Formule |
| :--- | :--- | :--- |
| `daily_return_pct` | Calculé | `((close - prev_close) / prev_close) * 100` |
| `ma_7`, `ma_20`, `ma_50`, `ma_200` | Calculé | Moyennes mobiles simples (SMA) calculées sur 7, 20, 50, et 200 jours ouvrés. |
| `rsi` | Calculé | Index de Force Relative (RSI sur 14 jours, méthode de lissage de Wilder). |
| `ma_signal` | Calculé | `BULLISH` si la MA_20 jours croise au-dessus de la MA_50. Sinon, défini comme `BEARISH`. |
| `price_z_score` | Calculé | Normalisation des prix : distance du prix de clôture par rapport à sa MA_200, en nombre d'écarts types. |

### C. Financials (Trimestriel & Annuel)

| Colonne | Source | Calcul / Formule |
| :--- | :--- | :--- |
| `revenue_growth_qoq_pct` | Calculé | `((revenue / prev_quarter_revenue) - 1) * 100` |
| `eps_growth_qoq_pct` | Calculé | `((eps - prev_quarter_eps) / ABS(prev_quarter_eps)) * 100` |
| `revenue_growth_yoy_pct` | Calculé | `((revenue / same_quarter_prev_year_revenue) - 1) * 100` |

---

## 3. Calculs en Temps Réel (`app.py`)

Les indicateurs de filtrage dynamiques calculés via l'application Python Streamlit (`get_master_screener_data`) :

| Colonne | Source | Calcul / Formule |
| :--- | :--- | :--- |
| `Quality Score (0-100)` | Calculé | Score composite évaluant les fondamentaux basé systématiquement sur la marge brute, ROE, FCF Margin et les ratios d'endettement. |
| `FMI Score (0-100)` | Calculé | Classement ("Percentile") de normalisation de l'élan fondamental. |
| `EPS Momentum` | Calculé | Marqué `Accelerating` si la croissance QoQ de l'EPS dépasse +10% sur les 2 derniers trimestres consécutifs. |

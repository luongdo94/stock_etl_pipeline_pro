# Honest Quant Intelligence Platform — Institutional Investment Review

As of 13 October 2026. Target: stock_etl_pipeline_pro-main-audited-r3.

## Scope, evidence and reliance

This is an investor and quantitative portfolio-management review, covering the five requested areas and their supporting ingestion, historical evidence, portfolio, alert and AI paths. It inventories the checkout and traces the investment-critical implementations; it is not a security certification or an assertion that every ancillary UI line was dynamically tested. Findings below come from this revision's source, not from previous audit conclusions. Only this report is added; no implementation changes are authorized or applied.

The source inventory contains 197 files; 181 Python, YAML, Markdown, text and SQL files were read into the inspection context, with detailed review concentrated on the investment-critical paths cited below. The checked-in warehouse contains only .gitkeep. There are no local prices, fundamentals, score snapshots, live fills or realized portfolio records to assess. Shell creation failed with helper_unknown_error/setup refresh errors; Python executables were unavailable through the alternate runtime. Consequently pytest, DuckDB end-to-end checks, live provider reconciliation and browser execution were NOT run. Existing test code was inspected; its presence is not proof of a passing run. Simple numerical examples below were recalculated independently in JavaScript, not executed through the Python implementation.

VERIFIED STATIC means the behavior is evidenced by the cited source. INFERENCE means the investment implication follows from that behavior. UNVERIFIED identifies an empirical claim or operational dependency that this checkout cannot establish. Scores are professional readiness judgments, not measured forecasts of future returns.

## 1. Executive verdict and institutional scorecard

**Research-use only. Do not authorize autonomous allocation, institutional order sizing or reliance on the displayed stops as capital-loss limits.** The platform is a useful discretionary screening and underwriting assistant with better disclosure and defensive logic than many retail dashboards. It does not yet demonstrate an investable, persistent alpha strategy or an integrated portfolio capital-protection system.

| Dimension | Score / 10 | Assessment |
| --- | ---: | --- |
| Alpha Potential | 4 | Economically credible factor families; custom thresholds, net portfolio alpha and capacity remain unvalidated. |
| Valuation Rigor | 5 | SBC deductions, scenario/reverse valuation, normalized cash flows and financial-sector routing are strengths; growth, financing, credit and payout economics remain incomplete. |
| Risk Protection | 3 | Useful risk diagnostics, freshness gates and stop-based sizing; no binding aggregate loss budget, liquidity-aware sizing or crash-responsive simulation. |
| Data Survivability | 4 | Shadow warehouse, continuity checks, adjustment rebases, prospective snapshots and vintage archives are valuable; provider independence, delisting outcomes and true historical as-of fundamentals are absent or unverified. |
| Execution Feasibility | 3 | Potentially useful for discretionary liquid large-cap orders; no executable spread, market-impact, participation, exit-capacity or short-borrow framework. |

The platform's likely benefit is avoiding some weak-business and stale-data purchases, improving comparability, and making assumptions inspectable. That is a process benefit. Persistent risk-adjusted alpha after trading costs, factor exposures and opportunity costs is **UNVERIFIED**. Large-cap Q/V/M exposure may earn returns without generating manager-specific alpha.

The operating design is predominantly long-only cash-equity decision support. Long/short mandates, market-neutral books, leverage, options overlays, borrow recalls and multi-strategy capital allocation are not supported by the inspected decision and optimizer paths.

## 2. Top three critical blind spots and proposed kill switches

### K1 — Investment labels can outrun the evidence and underwriting

**VERIFIED STATIC:** An unsupported track record adds one confidence note; it does not block BUY. One or two notes still produce MEDIUM confidence, which is permitted. Risk flags also add one aggregate note rather than category-specific vetoes. Momentum, estimate revisions and qualitative disruption risk are not inputs to the BUY gate. Evidence: [decision.py](../../core/decision.py#L154), lines 154–187 and 204–224; [screener.py](../../core/screener.py#L108), lines 108–112. The regression fixture explicitly permits a BUY with track_record_ok=False: [test_audit_round3.py](../../tests/test_audit_round3.py#L108), lines 108–115.

**Capital-loss mechanism:** A cheap-looking business can receive a sized BUY while the signal has no demonstrated predictive value, future earnings are structurally deteriorating, and critical negative evidence merely changes a confidence color. Factor scores do not constitute a variant-perception thesis or prove that cash flows will persist.

**Proposed kill switch:** Disable implementation-ready purchase and rebalance outputs until source reconciliation, material thesis falsifiers, liquidity capacity and mandate-specific risk constraints pass. Label unsupported opportunities RESEARCH CANDIDATE. Keep a separate evidence gate for the actual decision strategy; Quality IC alone cannot certify the DCF-plus-stop rule.

### K2 — Risk diagnostics do not constrain aggregate capital at risk

**VERIFIED STATIC:** All Monte Carlo paths share one deterministic sigma forecast; simulated adverse shocks never update their future variance. Stops determine size, while the Risk Lab explicitly does not feed the Decision. Portfolio analytics fill missing returns with zero; optimizers can allow 40% individual positions despite the decision engine's 10% cap. Evidence: [risk_range.py](../../core/risk_range.py#L63), lines 63–76; [risk_lab.py](../../views/risk_lab.py#L42), lines 42–46; [decision.py](../../core/decision.py#L77), lines 77–88; [portfolio.py](../../views/portfolio.py#L544), lines 544–575 and 963–1024.

**Capital-loss mechanism:** Correlated names can simultaneously gap through stops, liquidity can disappear, and an optimizer can concentrate exposures outside the candidate-sizing policy. A 10% position with a 10% stop targets 1% NAV loss at the stop; a 40% adverse gap loses 4% NAV before execution costs. Ten fully invested correlated positions undergoing that shock lose approximately 40% NAV. These are illustrative arithmetic, not measured probabilities.

**Proposed kill switch:** Veto additions and rebalance proposals that breach portfolio stress-loss, aggregate factor/sector concentration or stressed exit-capacity budgets. Missing return observations must fail data-quality eligibility rather than masquerade as zero-risk observations. Size to the tightest stress, liquidity and exposure constraint.

### K3 — Apparently precise intrinsic values can capitalize incomplete cash flows

**VERIFIED STATIC:** Missing SBC, financing interest and lease deductions become zero in the cash-flow calculation. The model grows this proxy for ten years without an explicit reinvestment, debt schedule or principal/net-borrowing bridge. Financial-company value depends on historical ROE and current book value without capital-adequacy or asset-quality checks. Evidence: [valuation.py](../../core/valuation.py#L160), lines 160–195, 198–209, 399–451 and 538–570; [extract.py](../../etl/extract.py#L1387), lines 1387–1402.

**Capital-loss mechanism:** Missing claims ahead of equity, a temporary working-capital cash release or under-provisioned bank profits are interpreted as durable distributable cash. The model can then manufacture an attractive margin of safety precisely when economic capital is deteriorating. The extractor itself says the financing-field mappings require live verification.

**Proposed kill switch:** Withhold an actionable valuation when material equity cash-flow deductions, debt-service requirements, dilution or bank capital/credit inputs cannot be verified. Unknown must remain distinct from confirmed zero. Require issuer-specific cash-flow or residual-income reconciliation before valuation drives a position.

## 3. Deep-dive critique

### A. Quantitative factors and alpha validation

**Strengths.** Quality, Value and Momentum are separated; analyst rating/target levels do not directly determine Q/V/M. Standard Quality weights are return on capital 20%, operating/gross/FCF margins 25% combined, growth/stability 20%, balance sheet 20%, and FCF conversion 15%. Value combines FCF yield, EV/EBITDA, earnings yield, realized-growth PEG, shareholder yield and P/S. Economic loss-makers receive observed low-value components rather than being treated as missing. [scoring_rules.yaml](../../config/scoring_rules.yaml#L32), lines 32–76; [scoring.py](../../core/scoring.py#L350), lines 350–363 and 428–442.

These factor families have a credible research basis: profitability, growth and safety feature in [Asness, Frazzini and Pedersen's Quality Minus Junk](https://www.aqr.com/Insights/Research/Working-paper/Quality-minus-Junk), and excluding the latest month resembles [Kenneth French's developed-market momentum construction](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_developed_mom.html). Neither source validates this repository's weights, band cutoffs, screens or net returns.

ROIC now uses EBIT times one minus tax divided by average invested capital, with historical medians and rejection of tiny/negative capital denominators. Diluted EPS improves PEG, and selected cyclicals use a longer earnings window and peak-margin flags. These current fixes deserve credit; the older net-income proxy is a fallback, not the sole implementation. [scoring.py](../../core/scoring.py#L154), lines 154–176 and 335–363.

**Flaws and limits.**

1. **Judgmental weights are not calibrated alpha.** Several Quality measures count related profitability information, while FCF appears in Quality, Value and DCF. Revenue growth and profitability also correlate. No inspected strategy-level evidence shows each component's incremental contribution, stability, turnover or cost-adjusted benefit. PEG rewards realized endpoint growth that may be nonrecurring; a three-year revenue CAGR does not establish competitive durability.
2. **Peer ranking reduces some comparability bias, not portfolio sector exposure.** The algorithm ranks against at least five valid curated reference names, falling back industry → sector → universe. With five distinct peers, mid-ranks are 10, 30, 50, 70 and 90: adjacent observations differ by 20 points. Fallback changes economic comparators and can change with metric availability. Blending absolute Value bands is useful when a whole sector is expensive, but it deliberately retains level differences; absolute ROIC/ROE/leverage and universe-wide Momentum also retain exposures. No sector/country/size/beta constraints make the selected portfolio neutral. [scoring.py](../../core/scoring.py#L68), lines 68–94, 380–404 and 473–504; scoring rules lines 13–18.
3. **Missingness can inflate a score.** Observed weights are renormalized, then S = 50 + (S_observed − 50) × min(1, coverage/0.8), before Quality penalties. Observed score 100 at 60% coverage becomes 87.5, rounded to 88. At 80% coverage, no shrinkage remains. Thus the full 20% balance-sheet component can be missing while the remaining favorable metrics produce full score credit. The BUY coverage threshold is only 60%; these are not posterior probabilities or validated conservative bounds. The missing list affects confidence as one note, not one veto per essential input. [scoring.py](../../core/scoring.py#L103), lines 103–124, 395–396 and 527–562; [decision_rules.yaml](../../config/decision_rules.yaml#L56), lines 56–60.
4. **Nominal coverage overstates evidentiary coverage.** A growth/stability component can count its full outer weight when only part of its inner inputs exists. ROIC uses a 21% default tax assumption when missing, and can use a legacy net-income proxy. Unobserved SBC is treated as zero while the reported FCF yield remains scoreable. None of these proxies causes coverage to fall in proportion to the uncertainty. [scoring.py](../../core/scoring.py#L154), lines 154–165, 282–299, 343–346 and 391–396.
5. **ROIC economic capital is simplified.** Equity + debt − all cash omits explicit operating-cash requirements, R&D capitalization, acquisition/goodwill treatment and some lease obligations. EV minus market cap is not always pure net debt: minority claims/preferred stock can enter EV. Precision of the ratio does not guarantee cross-company comparability. Scoring lines 301–315 and 338–346.

**IC, lag and decay.** The prospective framework is genuinely useful. Snapshots carry knowledge_at, score version, rule hashes and component/valuation inputs; INSERT OR IGNORE protects first-seen daily recommendations. Forward outcomes start at the first close strictly after max(as-of date, knowledge date). Daily Spearman IC receives Bartlett Newey–West errors with lag h−1. Horizons span 21/63/126/252 sessions, and the evidence gate requires positive Quality IC, t > 2, 60 usable days and six nominal nonoverlapping annual periods. [snapshot.py](../../etl/snapshot.py#L117), lines 117–139 and 143–158; [track_record.py](../../core/track_record.py#L55), lines 55–125 and 199–230.

However, no actual IC or decay curve is supplied here: **UNVERIFIED**. Six annual periods require roughly six years of signal origins with matured outcomes, and an additional year to mature the last annual cohort. The code computes independent-period count as floor(number of daily IC observations / horizon), not an explicit calendar-block audit. It does not adjust for searching across factors, horizons, bands and presets. Version filtering uses SCORE_VERSION alone despite storing rules_version; a YAML change without a score-version bump can mix different definitions. No decision-rule/code hash certifies all implementation changes. Evidence currently validates Quality, not the full BUY mechanism. [track_record.py](../../core/track_record.py#L187), lines 187–222; snapshot lines 32–40. Selection across many specifications needs explicit research controls, as discussed in [Bailey et al., The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb).

SPY-relative Spearman correlation does not establish factor-adjusted alpha: where every stock uses the same return window, subtracting a common benchmark return leaves cross-sectional ranks unchanged. Regional/currency/style exposure and changing sample membership remain. Action scorecards average overlapping observations rather than construct a funded portfolio with costs and risk constraints. [track_record.py](../../core/track_record.py#L128), lines 128–167.

The historical technical simulator flags its current-Quality look-ahead rule, shifts held-position returns to the next bar and charges explicit costs. Still, it decides from closing indicators and assumes execution at that same close, uses close-only stops, and cannot validate fundamental alpha. A final signal-day fee fix does not resolve closing-auction feasibility, gaps, market impact or survivor selection. [backtest.py](../../core/backtest.py#L45), lines 45–106 and 130–137.

### B. Fundamental valuation and value traps

**Strengths.** The DCF prefers the median of three annual statement FCF observations, deducts SBC where observed, avoids subtracting debt a second time from its equity-value proxy, and uses reverse DCF, growth/rate scenarios and sensitivities. It rejects some implausible values and terminal-value dependence above 80%. It separately routes banks/insurers to justified P/B and asset-light financial-data businesses to FCF valuation. [valuation.py](../../core/valuation.py#L182), lines 182–217, 316–334, 463–491 and 524–582.

Implemented equations are:

- CF_t = CF_(t−1) × (1 + g_t), with g_t fading linearly over ten years.
- Equity proxy value = Σ CF_t/(1+r)^t + CF_10 × (1+g_terminal)/[(r−g_terminal)(1+r)^10].
- r = currency risk-free rate + clipped Blume-adjusted beta × 5% ERP.
- Financial equity value/share = book/share × (ROE_sustainable−g)/(r−g).

**Flaws and limits.**

1. **FCFE is a proxy without financing economics.** CFO less capex, SBC and selected financing deductions does not reconcile net borrowing, mandatory principal repayments, refinancing requirements or leverage evolution. The appropriate debt cash-flow bridge depends on the chosen leverage policy; simply adding borrowing mechanically would not cure the problem. Reinvestment is already embedded in historical CFO/capex but not causally connected to forecast growth or returns on incremental capital. [Damodaran's cash-flow and growth definitions](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/definitions.html) show the missing linkage between debt needs, reinvestment and growth. Valuation lines 160–171 and 198–209.
2. **Missing adjustments are permissive.** Missing SBC/interest/lease data becomes zero. sbc_available and fcfe_adjustments require only any recent observation, so partial three-year coverage can read as adjusted. The source description warns about absent financing deductions, but reliability does not automatically fail on that absence. Live Yahoo-field reconciliation is explicitly unfinished in extract.py lines 1391–1402. A material omission can therefore pass valuation reliability.
3. **Normalization can preserve obsolete earnings power.** Three-year median cash flow damps spikes but can lag a permanent collapse. Forecast growth is bounded by past revenue plus clipped earnings/FCF history, not by explicit price/volume, margins, capex, working-capital or competitive scenarios. FCF conversion does not distinguish receivables factoring, supplier-finance extensions, tax timing or inventory liquidation. Valuation lines 119–152 and 538–567; scoring lines 282–299 and 404.
4. **Scenario uncertainty is too narrow to represent tail loss.** Bear/bull change starting growth ±5 percentage points and discount rate ±1 point, using the same cash-flow base and terminal growth. The 25/50/25 probabilities are judgments. Bankruptcy, covenant breach, adverse dilution and an abrupt cash-flow reset are absent. A probability-weighted average of scenario IRRs is not the IRR of expected cash flows and is not a measured investor expected return; the UI now correctly says so. Valuation lines 295–322 and 511–519; [decision_panel.py](../../ui/decision_panel.py#L94), lines 94–103.
5. **Rates are partly static, and country risk is not explicit.** Currency assumptions are directionally more coherent than discounting all cash flows at the US yield. Nevertheless only USD/HKD can use the live 10Y feed; other entries are configured approximations. The same 5% ERP and beta clipping do not explicitly capture country/sovereign risk, business mix, distressed financing or FX evolution. Terminal growth needs a sustainable reinvestment policy and terminal ROIC/ROE, not merely r > g. [decision_rules.yaml](../../config/decision_rules.yaml#L24), lines 24–54; valuation lines 50–97.
6. **Justified P/B needs a bank balance-sheet model.** Median ROE over up to four years, clipping and a 30% downward fade of excess returns improve on capitalizing peak ROE forever. But ROE uses closing equity, current book/share is inferred from price/PB, and no CET1, leverage ratio, tangible-book reconciliation, NPL, credit-loss/provision, deposit funding, duration or underwriting/reserve adequacy inputs constrain it. Four benign years may not cover a credit cycle. Asset-light exception logic does not replace credit analysis for lenders. Valuation lines 387–451.
7. **Quality ≥ 50 is only a coarse trap screen.** A financial score consists of ROE 40%, net margin 25% and growth/stability 35%; a highly leveraged bank can look excellent just before provisions normalize. Industrials may remain above the floor despite capped penalties or disappearing cash conversion. No inspected factor implements accrual decomposition, Beneish/Piotroski diagnostics, covenant/maturity risk or a structural-disruption veto. Scoring rules lines 34–58 and scoring lines 408–425.
8. **The margin-of-safety label is nonstandard but internally explicit.** Value/price−1 ≥25% means price is 20% below value, not 25% below value. A true 25% discount to value requires value/price−1 ≥33.3%. The current rule is internally consistent, but PMs must interpret the denominator correctly. Decision lines 201–224.

### C. Tail risk, position sizing and drawdown engineering

**Strengths.** GJR-GARCH with Student-t innovations is a reasonable conditional daily-volatility starting point. Percentage/decimal scaling is explicit; fallbacks are visible; VaR/ES are reported; calibration refits before each origin using nonoverlapping windows. The current UI requests up to 60 available windows and requires at least 30 for a calibration judgment with a Wilson interval. It is no longer restricted to eight windows despite stale checkbox text. [risk_range.py](../../core/risk_range.py#L25), lines 25–51 and 96–146; [risk_lab.py](../../views/risk_lab.py#L87), lines 87–92.

**Flaws and limits.**

- A GJR model's intended variance recursion is σ²_(t+1) = ω + αε²_t + γε²_t 1(ε_t<0) + βσ²_t. The fitter estimates asymmetric conditional volatility, but the simulation does not run that recursion on each simulated path. It draws independent shocks against the same sigma vector. A crash shock cannot increase that path's subsequent volatility. [arch's forecasting documentation](https://arch.readthedocs.io/en/latest/univariate/forecasting.html) distinguishes conditional variance forecasts from simulation that forward-evolves the fitted process. Source: risk_range.py lines 39–44 and 63–76.
- A 500-return fitting window and a cap at four times long-run daily standard deviation do not guarantee suitability during a liquidity freeze. The fit does not explicitly inspect convergence flags, stationarity/persistence constraints, residual dependence, parameter uncertainty or outlier/event diagnostics. Gaussian/fallback substitution can materially change tail behavior. Degree-of-freedom values at or below 2.05 silently use normal shocks in simulation. Lines 34–51 and 71–74.
- The implemented log increment is (mu − σ²/2)+σz, using volatility fitted to simple percentage returns. At default mu=0, Gaussian price paths have zero expected arithmetic drift and negative log drift; this is not zero log drift. With Student-t log shocks, the Gaussian compensator does not create a finite zero-drift expected price: a Student-t variable has no finite moment-generating function for a nonzero argument. Empirical lower-tail VaR and long-position loss ES can still be calculated; the mean-price interpretation is invalid and upper-tail simulations can be unstable. This is a mathematical implication of the exact distribution/exponential formula, not a live-data result. Lines 63–76.
- Zero drift avoids pretending historical mean return is a reliable short-term forecast. It is not inherently conservative for every equity state: it may overstate loss risk relative to a positive-drift baseline or understate a distress process with negative drift. Over longer horizons, dividend, FX, risk-premium, survival and regime assumptions matter. The UI's actual horizon is 5–90 sessions; it does not model a multi-year investment thesis. risk_lab.py line 58.
- risk_metrics measures terminal losses, not maximum interim drawdown, stop-breach probability, recovery time, liquidation loss or joint multi-asset tail events. Continuous strictly positive simulated prices have no bankruptcy/halt mass. Coverage inside an 80% range does not validate 95% ES severity or clustered VaR breaches. Lines 84–146. At the maximum 60 calibration windows, only about three 5% tail outcomes are expected; tail-loss estimates remain very noisy.
- Stop-based sizing is a conditional loss budget, not an absolute cap. Thesis stop is the wider technical/bear anchor and at least 8% below entry; suggested size is min(1% NAV risk / fractional stop distance, 10% NAV). The stop can move if recomputed from a new price/model unless an original plan is separately persisted. No execution or portfolio drawdown controller enforces it. Decision lines 77–88, 141–152, 261–290; alerts.py lines 50–81.
- Portfolio concentration is displayed after sizing, not enforced. candidate_impact omits cash/short exposures and zeros some missing observations. Current portfolio VaR/ES uses a historical fixed-weight basket and zero-filled returns. It is not realized transaction-ledger performance and may understate variance when missingness drives low covariance. [portfolio_risk.py](../../core/portfolio_risk.py#L21), lines 21–50; portfolio.py lines 535–575.
- Historical stress panels do exist: current SPY beta times fixed crisis index drawdowns. They are rough sensitivity illustrations, not replay of the holdings' crisis paths or credit/liquidity/FX stresses. Optimizers use sample covariance and 50% shrinkage of historical means; this is heuristic mean shrinkage, not a fitted James–Stein estimator or covariance shrinkage. Min-volatility/max-Sharpe permit 40% weights and risk parity permits up to 100%, without the Decision's 10% cap or loss budget. portfolio.py lines 651–678 and 963–1024.

The SEC explains that a stop price is a trigger rather than a guaranteed execution price: [Trading Basics](https://www.sec.gov/file/trading101basicspdf). A disclosed gap warning is useful; it does not substitute for stress-based sizing.

### D. Data integrity, microstructure and institutional capacity

**Strengths.** Prices request auto-adjustment; overlapping observations detect adjustment drift; weekly rebasing and minimum history-retention checks defend against splits/dividends and partial responses. Prospective outcome history stores returns calculated within a consistent adjustment basis rather than mixing historical price levels across rebases. EUR conversion retains FX and price-scale provenance, handles minor quote units and separates reporting from listing currency. Missing FX is often rejected rather than set to one. These are meaningful defenses. [extract.py](../../etl/extract.py#L436), lines 436–479, 584–610 and 672–756; [integrity.py](../../etl/integrity.py#L60), lines 60–124; snapshot.py lines 43–68.

Shadow release gates compare universe coverage, freshness and continuity before atomic promotion; per-stock BUY freshness gates supplement systemic checks. [gates.py](../../etl/gates.py#L39), lines 39–110; [pipeline.py](../../etl/pipeline.py#L422), lines 422–458. Annual statement and FCF vintages retain distinct payloads and first-received availability; prospective score history protects first knowledge.

**Flaws and limits.**

1. yfinance and yahooquery are two clients of the same Yahoo source, not independent corroboration. Retry/backoff reduces download failures but does not prove financial-field accuracy. [The yfinance maintainer's README](https://github.com/ranaroussi/yfinance/blob/main/README.md?plain=1) describes research/education use and directs users to Yahoo's data-use terms. Institutional rights, service commitments and a licensed independent fallback are **UNVERIFIED**; no legal determination is made here.
2. Prospective snapshots reduce signal look-ahead but do not solve survivor/delisting outcomes. The curated list is current; discovered names expire after 30 days, their raw history is garbage-collected, and transforms filter inactive tickers and zero-volume rows. The separate return archive preserves observations already seen, but no inspected delisting payoff, merger consideration, permanent security ID or explicit suspension valuation closes terminal outcomes. Missing forward returns are dropped from IC; problematic exits can therefore be excluded rather than charged as losses. [universe.py](../../etl/universe.py#L93), lines 93–124; [transform.py](../../etl/transform.py#L77), lines 77–80; track_record.py lines 84–92 and 117–120.
3. Vintages are not a complete historical as-of research engine. Historical financial marts select the latest loaded rows, while the vintage archive is not queried by the inspected scoring/backtest paths. FCF periods are reduced to year labels, then vintage period-end is proxied to December 31; availability uses a 90-day lag and first-received date, not actual intraday filing time. This is conservative for backdating but cannot recover what the market knew before collection began. load.py lines 154–197; extract.py lines 1366–1371; transform.py lines 304–338.
4. Append-only first-seen returns prevent adjustment-basis mixing but also preserve first-seen erroneous returns unless a governed correction mechanism exists. Outcome history failure and score-snapshot failure are nonfatal to ETL success. Evidence completeness needs its own production gate and exception ledger. Snapshot lines 52–59 and 169–175; pipeline lines 453–458.
5. FX price alignment uses forward fill and backfill; the latter can insert a future available FX observation into an earlier missing date. Historical statement amounts use extraction-time FX; cross-run historical euro amounts can change solely because FX changed. These policies need documented accounting/research uses, timestamped rates and a no-future-fill as-of mode. extract.py lines 293–314, 548–557, 597–600 and 947–974.
6. Liquidity is measured, but not enforced. ADV is actually median adjusted EUR close × volume over the last 60 observed sessions, not mean raw traded turnover. Below EUR 1m is informational; missing ADV is not flagged because NaN comparisons become false. A EUR 10bn market-cap BUY floor helps narrow the universe but says nothing about spread, free float or the chosen listing's trading capacity. Scoring lines 231–233 and 539–544; scoring rules lines 78–80; decision rules line 60. Scanner refinements have no ADV control: scanner.py lines 182–220.
7. No inspected order framework constrains bid/ask spread, volatility-dependent impact, participation, exit days, auction fills, short availability, borrow fees or market halts. Commission and FX costs are flat percentages. Annual withholding drag is subtracted once from a horizonless valuation upside, while the displayed four-year model IRR remains gross. These cannot establish net strategy economics. Decision lines 58–74 and 134–152; decision_panel.py lines 94–103.

**Illustrative capacity check:** EUR 1bn NAV at 10% weight implies EUR 100m in one security. At EUR 1m daily turnover and an assumed 10% participation ceiling, exit takes 1,000 trading days. This is not a forecast for any supplied stock, nor a recommended participation rate; it demonstrates why a fixed liquidity flag cannot certify institutional capacity. Real capacity is **UNVERIFIED** until NAV, venue, actual turnover, stress assumptions and trade schedule are specified.

**Deployment caveat:** arch is only in requirements-ml.txt; Docker installs requirements.txt alone, whereas CI explicitly installs arch. Without an additional install, the delivered default container uses the mean-reverting fallback, not the advertised fitted GJR model. The fallback is disclosed in the model caption, but production methodology must be tied to installed dependencies. Dependency lower bounds without a lock also weaken exact research reproducibility. [requirements-ml.txt](../../requirements-ml.txt#L1), lines 1–3; [Dockerfile](../../Dockerfile#L14); [.github/workflows/tests.yml](../../.github/workflows/tests.yml#L10).

### E. Decision architecture, signal conflicts and 30-second workflow

**Strengths.** Shared decide() consolidates scanner/panel logic. BUY requires reliable valuation, base cushion, reward/risk, non-low confidence, known sufficient quality/coverage/market cap and recent price/fundamentals/statement dates. Unknown quality, coverage and statement dates fail closed. The main panel fronts stance, confidence, model IRR, bear loss, reward/risk and suggested size, followed by reasons and invalidation. AI briefs are instructed to restate the deterministic stance, not invent a competing target. Decision lines 191–255 and 300–321; decision_panel.py lines 74–144; services/ai.py lines 324–342.

**Flaws and limits.**

- Confidence counts notes equally rather than assessing materiality. Five risk flags bundled into one note can count less than three mundane omissions. A critical credit concern should not be offset by having fresh metadata. Unvalidated evidence plus one liquidity/red-flag note can still be MEDIUM and BUY. Decision lines 154–187.
- Cheap valuation plus collapsing momentum is not explicitly reconciled in the recommendation. Momentum/revisions are context only; structural news can be described as a risk without changing deterministic stance. rating.py clamps an UNFAVOURABLE context to NEUTRAL when Decision says BUY. This makes labels cosmetically consistent but can conceal the actionable conflict. The pillar details remain visible; the conflict should be elevated as a distinct review state. [rating.py](../../core/rating.py#L251), lines 251–257; [signal_matrix.py](../../core/signal_matrix.py#L19), lines 19–21.
- AVOID/TRIM does not know whether a position is held, its current weight, tax lots or target weight. HOLD/WATCH similarly conflates maintain-existing with wait-to-initiate. They are triage labels, not executable trade instructions. Price-driven AVOID can still arise on stale/low-confidence data, because the BUY freshness gates do not govern that branch. Decision lines 223–255.
- Earnings blackout is advisory, not a veto; the scanner does not pass next_earnings to decide. The panel can show an event warning without resizing the position. Decision lines 274–278; screener lines 108–112.
- Gross multi-year IRR, horizonless net upside and stop-based reward/risk are different quantities. The caption now discloses this honestly, but a PM scanning the headline may still read them as one internally matched payoff model. Justified-P/B outputs do not provide a corresponding holding-period IRR, so financial names lose that headline comparator.
- The score strip substitutes neutral Quality for display when the true screener Quality is unknown; the actual scanner decision receives the raw None. Most fail-closed coverage checks prevent actionable use, but missing evidence should remain visually missing throughout. Screener lines 89–94 and 181–190; context.py lines 109–119.
- A QoQ acceleration preset reads YoY columns. The helper, presets and label say two consecutive quarterly growth readings above 10%, but its callers supply eps_growth_yoy_pct and revenue_growth_yoy_pct even though SQL defines separate QoQ columns. This is a verified investment-classification mismatch, not merely stale documentation. Screener lines 33–61; transform lines 315–320; scan_presets.py lines 61–63 and 88–90. Separately, two-quarter QoQ alone is highly seasonal and needs fiscal/industry context.
- Residual UI claims undermine the single-recommendation discipline: market_regime.py emits GOLDEN BUY from RSI/Quality alone; the docs still describe ROIC as a net-income proxy and label Track Record bias-free without sufficient delisting qualifications. portfolio.py attaches professional-manager analogies to optimizers that have not been independently validated. Market regime/price-volume pattern labels are heuristic context, not observed institutional flows or calibrated probabilities. [market_regime.py](../../core/market_regime.py#L69); [smart_money.py](../../core/smart_money.py#L1); [docs.py](../../views/docs.py#L45).

**30-second institutional target:** Show one action conditional on holdings, the binding constraint, priced base/bear/stress scenarios with a common horizon, maximum permitted weight, stressed exit days, unresolved material source items, explicit conflicts and the next thesis review event. Expandable technical diagnostics belong behind that decision view. Current layout is close to useful triage, but readiness and binding constraints are missing.

## 4. Actionable institutional roadmap

These are proposals, not changes applied during the audit. Sequence follows dependency and capital-risk severity.

| Priority | Enhancement | Concrete deliverable | Acceptance condition |
| --- | --- | --- | --- |
| P0 / 1 | Institutional source and historical evidence layer | Stable security IDs; filing/report/receipt timestamps; actual fiscal periods; retained universe membership; corporate-action/delisting payoffs; reconciled native and EUR facts; versioned corrections; independent provider cross-check; research dependency lock. | Reproduce any historical decision using only then-available facts and a full code/rules/universe version; every failed/merged/delisted security has a documented outcome or remains an explicit unresolved exposure. Material discrepancies veto actionable output. |
| P0 / 2 | One portfolio and execution risk policy | Stress/gap-aware sizing; ADV, spread, participation and stressed exit-day limits; factor/sector/FX/gross/net and aggregate stress-loss constraints; common limits for candidates and every optimizer; missing-return exclusions; actual holdings/cash/trade ledger. | Inject a correlated gap, stale quote, missing return, halted listing or liquidity collapse and obtain a blocked/reduced trade. Portfolio-wide constraints cannot be bypassed by optimizer or UI. Report realized P&L separately from hypothetical historical baskets. |
| P1 / 3 | Issuer-specific valuation and earnings-quality underwriting | Forecast revenue/margins, capex, working capital and incremental ROIC; debt/refinancing and dilution bridge; justified-P/B residual-income/capital model; accrual and cash-conversion decomposition; sector-appropriate Piotroski/Beneish diagnostic coverage; impairment and credit scenarios. | Tie the equity cash-flow bridge to primary reports. Distinguish unknown from zero deductions. Require a sustainable reinvestment/terminal-return identity and a credible stressed equity outcome. Diagnostics are flags, not universal fraud verdicts or automatic alpha. |
| P1 / 4 | Investable alpha research harness | Frozen hypotheses; as-of universe and realistic signal/auction lag; forward/replayed portfolios; walk-forward out-of-sample evaluation; factor/regional benchmark attribution; turnover, spread/impact/borrow costs; trial log and multiple-testing controls; factor decay and regime stability. | Demonstrate incremental net portfolio alpha beyond standard Q/V/M and simple equal-weight controls with uncertainty estimates, stable specifications and acceptable stress drawdown. Validate actual decision stances separately from Quality rank IC. Paper-trade with reconciliation before mandate deployment. |
| P1 / 5 | Calibrated tail model and conflict-aware PM workflow | Pathwise GJR recursion or filtered historical simulation; coherent return/drift mapping; jumps/event and joint crash scenarios; convergence/residual tests; VaR independence/coverage and ES severity checks; mandatory runtime method disclosure; holdings-aware actions and explicit conflict state. | Show expected method in the deployed container; validate lower-tail behavior on multiple crisis regimes and large samples; a cheap-but-deteriorating name triggers documented re-underwriting rather than label clamping. All displayed return/risk quantities have consistent horizon/cost definitions. |

Specific stop/add/trim/drawdown numbers should come from the fund mandate and empirical stress/capacity evidence; this report does not invent universal limits. Increasing score complexity before these five foundations would add precision without proving investment value.

## 5. Acceptance boundary and unresolved empirical questions

**Suitable now:** supervised idea triage, peer comparisons, assumption exploration and monitoring of sourced thesis inputs. **Requires independent underwriting:** every suggested purchase, trim and allocation. **Not established:** persistent net alpha, strategy Sharpe, realized maximum drawdown, crisis ES calibration, investable turnover/capacity, institutional data rights or a loss-limiting execution process.

Before capital authorization, the review committee needs production data/snapshot counts and their full versions, native statements for representative issuers, security lifecycle coverage, funded historical/paper-trading results, actual spread/fill/borrow evidence, source rights and a single enforceable mandate risk policy. The checked-in tests show thoughtful regression coverage but cannot answer those financial questions by themselves.

The central investment judgment is that the platform can improve research discipline, while its current scores and risk ranges do not establish capital protection or persistent alpha. Preserve its transparent safeguards; prioritize reliable evidence and binding portfolio constraints before adding more predictive labels.

## 6. Tái kiểm tra vòng 4 — nhận xét độc lập sau FIXES_ROUND4

**Ngày rà soát:** 13/10/2026. **Phạm vi:** K1–K3, hai lỗi snapshot/DuckDB và các test liên quan trong vòng 4. Đây là phần bổ sung cho báo cáo gốc ở trên; các mô tả trước sửa trong phần 1–5 không nên được hiểu là trạng thái hiện tại khi phần này ghi nhận thay đổi tương ứng.

### 6.1. Kết luận và giới hạn kiểm chứng

Các thay đổi vòng 4 đã cải thiện đáng kể cơ chế chặn giao dịch, nhưng **chưa thể đóng hoàn toàn K2/K3**. Ba điểm dưới đây vẫn sửa được trực tiếp bằng code; cần phân biệt chúng với các hạng mục còn mở do thiếu hạ tầng hoặc dữ liệu.

Đã xác nhận bằng đọc source: K1 có nhánh RESEARCH CANDIDATE không đề xuất tỷ trọng; có veto cờ đỏ và xung đột momentum/earnings. Biến _DDL đã được định nghĩa trong etl/snapshot.py. requirements.txt đã thêm arch vào runtime chính và giới hạn duckdb>=1.4.4,<1.5. Đây là xác nhận triển khai trong source, không phải nghiệm thu hoạt động production.

Lỗi _DDL là điểm người rà soát đã bỏ sót trong audit trước. Việc định nghĩa lại sửa nguyên nhân NameError được báo cáo; vẫn cần quan sát một ETL thực để xác nhận snapshot được lưu, mirror và tích lũy qua nhiều lần chạy.

**421 passed, 14 deselected (live)** là kết quả do người triển khai báo cáo trong audit/fixes/FIXES_ROUND4.md, gồm smoke test Streamlit và 26 test mới. Người rà soát đã đọc code/test nhưng **chưa tái chạy độc lập**: thử interpreter tại C:/Users/luong/AppData/Local/Temp/hqv/Scripts/python.exe bị lỗi Access is denied khi truy cập Python nền. Không suy diễn lỗi này thành thất bại của bộ test. Live provider reconciliation, ETL dữ liệu thật và kiểm tra trình duyệt vẫn chưa được nghiệm thu. Nguyên nhân tương thích DuckDB 1.5.0 chưa được tái hiện độc lập trong lần rà soát này.

### 6.2. R4-01 — Trần optimizer có thể vượt 10%; rebalance chưa đi qua eligibility của Decision

**Mức ưu tiên:** P1. **Trạng thái:** OPEN — có bằng chứng tĩnh trong code và test.

**Files/functions:**

- core/portfolio_risk.py:38 — optimizer_weight_cap().
- views/portfolio.py:566 — render(), nơi lấy trần từ Decision và gọi helper.
- views/portfolio.py:1068 — render(), luồng GENERATE OPTIMAL REBALANCE.
- views/portfolio.py:1182 — render(), luồng APPLY REBALANCE.
- tests/test_audit_round4.py:134 — test_optimiser_cap_follows_the_decision_position_cap().

Helper trả về max(configured_cap, 1/n_assets). Với trần cấu hình 10%, bốn cổ phiếu được nâng lên 25% mỗi mã; một cổ phiếu được nâng lên 100%. Test hiện tại yêu cầu kết quả 25%, nên đang bảo vệ ngoại lệ này, không chứng minh trần 10% được giữ tuyệt đối. UI có giải thích ngoại lệ, nhưng thông báo không thay thế thực thi giới hạn.

Đường Generate Rebalance tạo BUY/SELL từ chênh lệch tỷ trọng mà không kiểm tra eligibility của Decision. APPLY cập nhật danh mục trong app; đây không phải bằng chứng khớp lệnh broker. Tuy vậy, chặn BUY ở scanner chưa chặn mọi đề xuất tăng vị thế trong sản phẩm.

**Hệ quả đầu tư:** Danh mục ít mã có thể được đề xuất mức tập trung vượt chính sách; một đề xuất tăng vị thế có thể xuất hiện dù mã đó chỉ đủ điều kiện nghiên cứu hoặc chưa đạt các cổng giao dịch.

**Sửa cụ thể:**

1. Giữ max_position_pct là trần tuyệt đối. Khi không đủ mã để đầu tư hết NAV, cho phép phần dư bằng tiền mặt hoặc báo bài toán không khả thi; không tự nâng trần để ép tổng tỷ trọng cổ phiếu bằng 100%.
2. Dùng một kiểm tra chung trước cả Generate và Apply cho mọi đề xuất tăng vị thế: evidence, valuation, liquidity và tổng tỷ trọng sau giao dịch. Tái kiểm tra dữ liệu/cấu hình hiện tại khi Apply, không chỉ tin đề xuất đã lưu trong session.
3. Các cổng mua áp dụng cho phần tăng vị thế; giảm vị thế cần chính sách riêng, không bị cổng evidence cho mua vô tình cản trở.

**Tiêu chí nghiệm thu:** Bốn cổ phiếu có tỷ trọng tối đa 10% mỗi mã và phần dư được thể hiện là tiền mặt, hoặc optimizer trả trạng thái không khả thi. Mã chưa đạt eligibility không được tăng vị thế qua bất kỳ optimizer nào. Test cần kiểm tra bất biến giới hạn, thay vì yêu cầu helper nâng trần lên 25%.

### 6.3. R4-02 — Cổng FCFE chưa yêu cầu đầy đủ khấu trừ và ba năm dữ liệu

**Mức ưu tiên:** P1. **Trạng thái:** OPEN — có bằng chứng tĩnh trong code và test.

**Files/functions:**

- core/valuation.py:183 — fcfe_adjustments().
- core/valuation.py:202 — sbc_available().
- core/valuation.py:591 — valuation_inputs(), cổng require_verified_fcfe_deductions.
- tests/test_audit_round4.py:175 — test_ifrs_filer_needs_verified_financing_deductions().

Cổng IFRS dùng điều kiện not adj. Chỉ cần danh sách có một loại khấu trừ được xác minh là cổng này không chặn nữa. Test tại dòng 179 yêu cầu reliable=True với lãi vay đầy đủ ba năm nhưng không có dữ liệu lease principal. Vì vậy, trạng thái lease chưa biết vẫn có thể đi qua cổng khấu trừ.

Ngoài ra, sbc_available() chỉ kiểm tra tập dữ liệu không rỗng và tất cả hàng hiện có đều có SBC. Một hoặc hai năm có SBC đầy đủ cũng có thể trả True. fcfe_adjustments() tương tự không yêu cầu đủ ba kỳ. Đây là khoảng cách giữa mô tả “đủ cả ba năm” và điều kiện thực thi.

_after_sbc() vẫn dùng fillna(0) để tạo giá trị tham khảo. Cách làm này chỉ an toàn cho ra quyết định nếu cổng reliability xác minh đúng mọi khoản áp dụng; các lỗ hổng nêu trên khiến cổng chưa đáp ứng điều đó.

**Hệ quả đầu tư:** Một định giá chưa phản ánh đủ nghĩa vụ đứng trước cổ đông có thể được đánh dấu đáng tin và trở lại làm cơ sở mua/bán.

**Sửa cụ thể:**

1. Kiểm tra riêng SBC, lãi vay trong financing và lease principal cho từng kỳ làm cơ sở định giá.
2. Mỗi khoản có trạng thái OBSERVED, CONFIRMED_ZERO, NOT_APPLICABLE hoặc UNKNOWN cùng bằng chứng nguồn. Không yêu cầu một khoản không áp dụng, nhưng phải xác minh tính không áp dụng; không suy ra từ trường bị thiếu.
3. Nếu chính sách yêu cầu ba năm, kiểm tra ba kỳ tài chính phân biệt với giá trị hợp lệ; không chỉ dùng .all() trên số hàng hiện có. Chọn kỳ theo quy tắc rõ ràng, tránh đủ ba hàng nhưng trùng kỳ hoặc bỏ qua một kỳ thiếu dữ liệu.
4. Bất kỳ nghĩa vụ áp dụng còn UNKNOWN phải giữ valuation unreliable, dù giá trị tham khảo vẫn được hiển thị. Quốc gia chỉ là heuristic, không phải bằng chứng về chuẩn kế toán hay cách phân loại dòng tiền của một issuer.

**Tiêu chí nghiệm thu:** Thiếu lease nhưng đủ interest vẫn bị chặn nếu lease chưa được xác minh là không áp dụng/zero; chỉ có một hoặc hai năm SBC bị chặn theo chính sách ba năm; reported zero hoặc NOT_APPLICABLE có bằng chứng được xử lý đúng. Test hiện tại cho qua trường hợp interest-only cần thay đổi.

### 6.4. R4-03 — Cắt riêng đuôi âm làm mô phỏng không còn martingale chính xác

**Mức ưu tiên:** P2. **Trạng thái:** OPEN — chứng minh trực tiếp từ công thức; độ lớn sai lệch trên dữ liệu thật UNVERIFIED.

**Files/functions:**

- core/risk_range.py:80 — simulate_paths().
- core/risk_range.py:111 — ret = np.maximum(mu_log_daily + eps, -MAX_LOSS_PER_DAY), với MAX_LOSS_PER_DAY=0.95.
- tests/test_audit_round4.py:101 — test_zero_drift_paths_are_a_martingale_even_with_fat_tails().

Chuyển sang simple returns đã loại bỏ vấn đề kỳ vọng vô hạn của exp(Student-t), và GJR đã được tiến hóa theo từng đường mô phỏng. Tuy nhiên, cắt lợi suất âm tại -95% tạo drift dương. Với shock R có kỳ vọng bằng không:

    E[max(R, -0.95)] = E[R] + E[(-0.95 - R)^+] > 0

Bất đẳng thức nghiêm ngặt khi phân phối còn xác suất dưới -95%, như normal hoặc Student-t với sigma dương. Do đó, công thức hiện tại không giữ kỳ vọng giá chính xác bằng giá hiện tại. Sai lệch có thể nhỏ trong điều kiện thường, nhưng tuyên bố martingale tuyệt đối không đúng, và phần tổn thất đuôi bị thay đổi khi sàn bắt đầu tác động đáng kể.

Test hiện tại dùng volatility 2% và dung sai trung bình 0.5%; chưa kiểm tra chế độ stress mà việc cắt đuôi có ảnh hưởng đáng kể. Một test Monte Carlo không bác bỏ được đồng nhất thức toán học trên chỉ vì mẫu/dung sai chưa phát hiện sai lệch.

**Sửa cụ thể:**

1. Thiết kế phân phối lợi suất thỏa giới hạn tổn thất và hiệu chỉnh kỳ vọng có điều kiện ngay trong mô hình; không coi phép cắt một phía là trung tính.
2. Nếu giữ giới hạn số học, công bố tỷ lệ đường/ngày bị giới hạn và ảnh hưởng lên kỳ vọng/tail metrics. Tách giả định bankruptcy/default khỏi guard giữ giá dương.
3. Thêm test stress cho drift, tần suất chạm sàn, độ nhạy VaR/ES và sự nhất quán giữa shock dùng cập nhật phương sai với lợi suất áp dụng vào giá. Không dùng tolerance rộng ở volatility thấp làm bằng chứng martingale cho mọi tham số.

**Tiêu chí nghiệm thu:** Kỳ vọng có điều kiện của multiplier được chứng minh/kiểm tra theo phân phối sau xử lý; test stress không che sai lệch bằng dung sai tùy ý; mô hình công bố rõ mọi giới hạn đuôi và tác động của chúng.

### 6.5. Trạng thái nghiệm thu sau tái kiểm tra

| Hạng mục | Trạng thái của lần rà soát này |
| --- | --- |
| K1 — research-only khi thiếu evidence, veto và momentum/earnings conflicts | Đã xác nhận triển khai trong code; evidence vẫn chỉ xác nhận Quality, và rebalance còn đường đề xuất ngoài eligibility. |
| K2 — bỏ zero-fill, pathwise GJR, kiểm tra thanh khoản và NAV-aware capacity | Đã có cải thiện; trần optimizer không tuyệt đối, drift sau cắt đuôi chưa đúng, ngân sách stress-loss toàn danh mục còn mở. |
| K3 — ngân hàng không dùng P/B thiếu capital inputs, provider FCF không điều khiển giao dịch | Đã xác nhận cổng tương ứng trong code; cổng đầy đủ khấu trừ/kỳ FCFE vẫn hở theo R4-02. |
| Snapshot _DDL và dependency requirements | Đã xác nhận thay đổi source; cần ETL thật/Docker build để nghiệm thu vận hành. |
| 421 passed / smoke tests | Kết quả người triển khai báo cáo; chưa tái chạy độc lập vì runtime bị Access is denied. |
| Alpha bền vững, dữ liệu nguồn, delisting, capacity và bảo vệ drawdown thực tế | UNVERIFIED; không thể suy ra từ bộ test phần mềm. |

Giữ các công tắc bảo thủ hiện tại trong khi xử lý R4-01–R4-03. Không nới cổng chỉ để xuất hiện thêm BUY. Bổ sung này chỉ ghi nhận kết quả review và đề xuất; không có sửa đổi implementation trong lần rà soát.

## 7. Tái kiểm tra Follow-up — R4-01 còn hai lỗ hổng rebalance

**Ngày rà soát:** 14/10/2026. **Phạm vi:** các sửa đổi Follow-up trong audit/fixes/FIXES_ROUND4.md và đường code/test của R4-01, R4-02, R4-03. Phần này bổ sung trạng thái mới; mục 6 được giữ nguyên như lịch sử nhận xét trước sửa.

### 7.1. Kết luận và giới hạn kiểm chứng

**R4-02 và R4-03 đã xử lý đúng nguyên nhân nêu trong mục 6 ở cấp code/công thức. R4-01 chưa thể đóng: còn hai lỗ hổng trên đường rebalance.**

Người rà soát đã đọc Follow-up, source và các test mới. **432 passed, 14 deselected (live)**, gồm smoke test Streamlit, vẫn là kết quả người triển khai báo cáo; chưa được tái chạy độc lập trong lần rà soát này. Các phản ví dụ số học dưới đây được tính độc lập từ đường source, không phải kết quả thực thi Python hoặc browser.

| Mục | Kết quả rà soát Follow-up |
| --- | --- |
| R4-01 | weight_budget giữ trần tuyệt đối và rebalance_policy kiểm tra eligibility ở Generate/Apply. Tuy nhiên, cash không được lưu vào NAV sau Apply, và policy chưa thực thi giới hạn tỷ trọng riêng theo exit capacity/stop-risk budget. OPEN ở cấp tích hợp. |
| R4-02 | Đã kiểm tra riêng từng khoản khấu trừ, yêu cầu ba năm tài chính liên tiếp/phân biệt và bỏ suy đoán theo quốc gia. Cổng dữ liệu đã xử lý nguyên nhân trước; nghiệm thu dữ liệu nguồn/issuer vẫn còn mở. |
| R4-03 | Multiplier được hiệu chỉnh bằng 1+c(σ), giải quyết drift do cắt đuôi về mặt toán học ở drift mặc định bằng không. Các test tích phân/stress phù hợp với lỗi trước; chưa xác nhận calibration trên dữ liệu thị trường thật. |
| Bộ test và hoạt động production | 432 passed là kết quả được báo cáo, chưa kiểm chứng độc lập. Live reconciliation, ETL thực và browser acceptance không được suy ra từ kết quả này. |

Với R4-03, khi c(σ)=E[max(ε,−F)] và E[ε]=0, multiplier M=(1+max(ε,−F))/(1+c(σ)) thỏa E[M|σ]=1. Việc cập nhật GJR bằng lợi suất đã áp vào giá và công bố ảnh hưởng của sàn là thay đổi đúng hướng. Điều này không tự chứng minh mô hình mô tả đúng volatility/tail risk của thị trường.

### 7.2. R4-F01 — CASH biến mất khỏi NAV sau Apply

**Mức ưu tiên:** P1. **Trạng thái:** OPEN — có bằng chứng tĩnh trên đường Apply → rerun.

**Files/functions:**

- views/portfolio.py:1099 — render() / helper _rebalance_table(), tạo dòng CASH trong bảng đề xuất.
- views/portfolio.py:1216 — render(), APPLY REBALANCE loại dòng CASH khi dựng new_shares_dict và chỉ lưu cổ phiếu vào session.
- views/portfolio.py:540 — render(), total_p_val được tính từ tổng Market Value của các cổ phiếu.
- views/portfolio.py:557 — render(), tỷ trọng hiện tại lấy giá trị cổ phiếu chia total_p_val.

Dòng CASH hiện có trong bảng đề xuất và thông báo Apply, nhưng không có số dư tiền mặt được lưu tương ứng trong đường Apply. Ở lần rerun, NAV và weights lại được dựng chỉ từ cổ phiếu. Vì vậy phần tiền mặt đề xuất không được bảo toàn trong trạng thái danh mục.

**Phản ví dụ từ source, giả định giá không đổi:**

1. NAV trước Apply là EUR 1.000.
2. Bốn cổ phiếu được phân bổ EUR 100 mỗi mã; EUR 600 còn lại là cash.
3. Apply chỉ lưu cổ phiếu có tổng giá trị EUR 400.
4. Rerun tính total_p_val=EUR 400, nên mỗi mã trở thành 25% thay vì 10% NAV.
5. Rebalance tiếp có thể tiếp tục thu nhỏ phần cổ phiếu vì EUR 600 không còn trong mẫu số NAV.

Đây là lỗi kế toán và giới hạn danh mục trong app; không phải bằng chứng tiền thật đã mất tại broker. Nó làm sai trọng số, quy mô danh mục và các phép tính risk/capacity sau Apply, dù helper đã giữ trần đúng trên NAV đầu vào.

**Sửa cụ thể:**

- Lưu số dư cash cùng holdings, cả trong session và cơ chế lưu/tải danh mục.
- Dùng NAV = equity market value + cash xuyên suốt sizing, analytics, Generate và Apply. Phân biệt rõ tỷ trọng trên NAV với tỷ trọng riêng trong phần cổ phiếu nếu cần hiển thị cả hai.
- Khi Apply, bảo toàn tài sản theo giả định khớp lệnh của app, bao gồm cash phát sinh từ bán/mua và chi phí nếu có; không bỏ phần dư khi dựng trạng thái mới.

**Tiêu chí nghiệm thu:** Một test tích hợp Apply → rerun → reload giữ NAV EUR 1.000, cash EUR 600 và mỗi mã 10% khi giá không đổi. Apply lại cùng đề xuất không được tiếp tục làm giảm tổng tài sản hoặc tái chuẩn hóa bốn mã thành 25%. Test helper weight_budget/rebalance_policy riêng lẻ không đủ chứng minh bất biến này.

### 7.3. R4-F02 — Rebalance có thể vượt exit capacity và giới hạn sizing của Decision

**Mức ưu tiên:** P1. **Trạng thái:** OPEN — có bằng chứng tĩnh trên giao diện policy và các đường gọi.

**Files/functions:**

- core/portfolio_risk.py:50 — rebalance_policy(), nhận stances và max_position_pct nhưng không nhận giới hạn tỷ trọng riêng của từng mã.
- views/portfolio.py:571 — render(), _stances chỉ giữ nhãn Decision theo ticker.
- views/portfolio.py:1123 — render(), Generate gọi policy với stance và trần chung.
- views/portfolio.py:1213 — render(), Apply gọi lại policy với stance và trần chung.
- core/decision.py:92 — liquidity_cap_pct(), tính giới hạn exit capacity khi NAV được cấu hình.
- core/decision.py:102 — position_size(), kết hợp stop-risk budget, trần vị thế và exit capacity cho Decision.

Nhãn BUY CANDIDATE xác nhận một mã qua các cổng eligibility, nhưng không mang theo giới hạn khối lượng được phép. rebalance_policy hiện có thể tăng mã này lên trần chung 10%, dù Decision đã giới hạn tỷ trọng thấp hơn vì thanh khoản hoặc stop-risk budget. Việc kiểm tra lại stance tại Apply chưa khắc phục việc đánh mất giới hạn sizing.

**Phản ví dụ số học:**

- NAV = EUR 1 tỷ.
- ADV = EUR 10 triệu; participation = 10%; exit tối đa = 5 ngày.
- Exit capacity = EUR 10 triệu × 10% × 5 / EUR 1 tỷ = 0,5% NAV.
- Policy rebalance vẫn có thể cho một mã có stance BUY CANDIDATE lên 10% NAV.

Ví dụ này giả định mã qua các cổng mua khác; nó không khẳng định một issuer cụ thể đang có các thông số đó. Khoảng trống nằm ở việc rebalance không thực thi giới hạn tỷ trọng mà sizing của Decision đã tính.

**Sửa cụ thể:**

- Truyền các giới hạn tỷ trọng riêng theo ticker vào policy, không chỉ nhãn Decision.
- Tính giới hạn tổng vị thế được phép bằng holdings/NAV hiện tại và kết hợp trần mandate, exit capacity, stop-risk budget; xác định rõ cách xử lý vị thế sẵn có đang vượt giới hạn.
- Kiểm tra lại trước cả Generate và Apply. Nếu NAV, ADV, stop hoặc quy tắc thay đổi, tính lại giới hạn bằng dữ liệu hiện tại thay vì dùng sizing cũ.
- Vẫn cho phép giảm vị thế; cổng mua không được vô tình cản trở việc giảm rủi ro.

**Tiêu chí nghiệm thu:** Trường hợp NAV/ADV ở trên không được đề xuất tăng vị thế lên quá 0,5% NAV. Test cần kiểm tra cùng giới hạn ở Generate và Apply, cùng tình huống NAV/ADV thay đổi giữa hai bước. Bổ sung trường hợp stop-risk budget thấp hơn trần chung để chứng minh policy không chỉ kiểm tra exit capacity.

### 7.4. Trạng thái bàn giao

R4-01 cần vượt qua hai test tích hợp nêu trên trước khi được coi là nghiệm thu toàn bộ đường Generate–Apply–rerun. Có thể ghi R4-02/R4-03 là đã xử lý nguyên nhân trước ở cấp code/công thức, đồng thời giữ trạng thái nghiệm thu độc lập/production còn mở.

Giữ nguyên mục 6 và các công tắc bảo thủ. Không suy ra alpha bền vững, calibration thực tế hay khả năng bảo vệ drawdown từ số lượng test phần mềm. Lần rà soát này chỉ bổ sung nhận xét vào báo cáo; không sửa implementation.

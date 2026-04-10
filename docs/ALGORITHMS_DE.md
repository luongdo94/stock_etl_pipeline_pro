# Dokumentation der Algorithmen & Indikatoren (System Algorithms)

Dieses Dokument erläutert die Berechnungslogik hinter den intelligenten Indikatoren auf dem Dashboard des Stock-ETL-Pipeline-Systems. Alle Daten im System (Preise, Umsatz, Marktkapitalisierung) werden bereits im Extraktionsschritt der ETL-Pipeline in **Euro (EUR)** normalisiert.

---

## 1. Trend Confidence Score & Market Regime
Dieser Indikator bewertet die Gesundheit des allgemeinen Markttrends und dient zur Bestimmung der Portfolio-Ausrichtung (Portfolio Stance). Dieser Score (0-100) misst die Stärke des allgemeinen Markttrends, wobei der Fokus auf dem US-Markt (S&P 500) liegt, aber zum Zwecke der Systemkonsistenz in **Euro (EUR)** umgerechnet wird.

### 1.1. Berechnung của Trend Confidence Scores (Max. 100 Punkte)

| Faktor | Bedingung | Gewichtung |
| :--- | :--- | :--- |
| **SPY Medium-term** | SPY-Schlusskurs > MA50 | +25 Punkte |
| **SPY Long-term** | SPY-Schlusskurs > MA200 | +25 Punkte |
| **Marktbreite (Breadth)** | % der Aktien im Universum > MA50 liegt über 50% | +30 Punkte |
| **Makro-Ausrichtung** | Makro-Status (VIX, DXY, TNX) ist `RISK_ON` | +20 Punkte |

*Hinweis: Wenn der Makro-Status `NEUTRAL` ist, werden nur +10 Punkte hinzugefügt.*

### 1.2. Klassifizierung des Market Regimes
Basierend auf dem Gesamtwert von `conf_score_global` klassifiziert das System den Markt in 4 Szenarien:

- **STRONG BULLISH ($\ge 75$):** Starker Aufwärtstrend, hohe Marktkonsistenz.
- **BULLISH ($\ge 50$):** Bestätigter Aufwärtstrend, geringes Risiko.
- **NEUTRAL / SIDEWAYS ($\ge 35$):** Seitwärtsmarkt, kein klarer Trend erkennbar.
- **BEARISH / CAUTION ($< 35$):** Markt schwächelt, hohes Risiko.

---

## 2. Quality Index & Individual Quality Score
Dieser Index repräsentiert die "intrinsische Qualität" des Marktes oder einer bestimmten Aktie.

### 2.1. Market Quality Index (Marktindex)
Alle Daten im System (Preise, Umsatz, Marktkapitalisierung) werden bereits im Extraktionsschritt der ETL-Pipeline in **Euro (EUR)** normalisiert.
`Market Quality Index = Σ(Quality Score * Market Cap) / Σ(Market Cap)`

### 2.2. Individual Quality Score v3.0 (100-Punkte-Skala)
Jede Aktie wird anhand von 6 finanziellen Säulen (Pillars) bewertet:
1.  **Valuation (20 Pkt):** Bewertung von KGV (P/E), KBV (P/B) und PEG (niedriges PEG bevorzugt).
2.  **Profitability (25-30 Pkt):** Fokus auf FCF-Marge und ROE.
3.  **Financial Health (15 Pkt):** Verhältnis Schulden/EBITDA basierend auf branchenspezifischen Merkmalen.
4.  **Net Payout Yield (10 Pkt):** Gesamtrendite für Aktionäre (Dividenden + Aktienrückkäufe).
5.  **Context & Momentum (25 Pkt):** Technische Signale (MA), relative Stärke (RSI) und Preisabweichung (Z-Score).
6.  **Analyst Estimates (5 Pkt):** Erwartetes Kurspotenzial (Upside) und Expertenkonsens.

**Strafen (Red Flags):** Punktabzug bei negativem KGV, Schulden/EBITDA > 10 oder sehr hohem Beta (>1.8).

---

## 3. Fundamental Momentum Index (FMI) v4.0
FMI misst die **Beschleunigung** (Acceleration) der Fundamentaldaten. Eine Aktie kann einen niedrigen Quality Score (aufgrund hoher Bewertung), nhưng einen sehr hohen FMI (aufgrund eines explosiven Wachstums) haben.

### 3.1. FMI-Scoring-Struktur (Max. 100 Punkte)
1.  **Revenue Acceleration (30 Pkt):** Umsatzwachstumsrate des letzten Quartals im Vergleich zum Jahresdurchschnitt.
2.  **EPS Acceleration (30 Pkt):** Wachstumsrate des Gewinns pro Aktie (EPS).
3.  **Margin Expansion (25 Pkt):** Ausweitung der Gewinnmargen (EPS wächst schneller als der Umsatz).
4.  **Earnings Consistency (15 Pkt):** Anzahl der Quartale mit positivem Wachstum in các letzten 4 Quartalen.

---

## 4. AI Trading Signature (Executive Verdict)
Dieses System befindet sich im Tab **Predictive Suite** und liefert die endgültige Handelsentscheidung basierend auf der Konvergenz von KI-Daten, Geldfluss und Risikomanagement.

### 4.1. Überzeugungswert (Conviction Score) - Skala von 0 bis 3
Der Score berechnet sich aus dem Konsens von 3 "Säulen":
1.  **AI Upside:** Modellprognose (LSTM/Transformer/PatchTST) $\ge 3\%$. (+1 Pkt)
2.  **Smart Money:** OBV ROC Indikator zeigt Akkumulation (Accumulation). (+1 Pkt)
3.  **News Sentiment:** Nachrichtenstimmung von FinBERT $> 0.05$. (+1 Pkt)

### 4.2. Risikomanagement R/R (Risk/Reward)
- **Reward:** Abstand vom aktuellen Preis zum KI-Kursziel (`_ai_target`).
- **Risk:** Abstand vom aktuellen Preis zum statistischen Stop-Loss (`_ai_stop` - 10. Perzentil der Monte-Carlo-Simulation).
- **R/R Ratio:** `Reward / Risk`.

### 4.3. Entscheidungslogik (Action Hierarchy)
Das System kombiniert den **Conviction Score** UND das **R/R-Verhältnis**, um das Urteil (Verdict) zu fällen:

- **STRONG LONG:** 3/3 Überzeugungspunkte UND R/R $\ge 1.5$.
- **BUY / ACCUMULATE:** $\ge 2/3$ Überzeugungspunkte UND R/R $\ge 1.0$.
- **REDUCE / HEDGE:** Wenn die KI einen Rückgang von $\le -3\%$ prognostiziert.
- **AVOID / WAIT:** Bei 0/3 Überzeugungspunkten (Widersprüchliche Signale).
- **NEUTRAL / MONITOR:** Alle anderen Fälle (Gemischte Signale).

---

## 5. Portfolio-Optimierungsstrategien

Das System bietet drei Optimierungsstrategien im Tab **Portfolio Builder** an, die es Anlegern ermöglichen, die Kapitalallokation basierend auf Risikoneigung und Diversifikationszielen anzupassen.

### 5.1. Max Sharpe (Markowitz MVO)
- **Prinzip:** Basierend auf der **Modernen Portfoliotheorie (MPT)** sucht das System nach den Gewichtungen ($w$), die die **Sharpe-Ratio** maximieren:
  $$\text{Sharpe-Ratio} = \frac{R_p - R_f}{\sigma_p}$$
- **Merkmale:** Konzentriert das Kapital auf die Vermögenswerte mit der besten risikobereinigten Rendite (hohe Rendite pro Risikoeinheit).
- **Eignung:** Für Anleger, die maximale Rendite anstreben und eine höhere Konzentration in Top-Performern akzeptieren.

### 5.2. Risikoparität (Risk Parity)
- **Prinzip:** Allokiert das Kapital so, dass jeder Vermögenswert einen **gleichen Risikobeitrag** zum Gesamtportfolio leistet. Das System löst das Optimierungsproblem:
  $$\min \sum_{i=1}^{n} (RC_i - \frac{1}{n})^2$$
  Wobei der Risikobeitrag ($RC_i$) definiert ist als: $RC_i = \frac{w_i (\Sigma w)_i}{\sqrt{w^T \Sigma w}}$
- **Merkmale:** Vermögenswerte mit hoher Volatilität erhalten weniger Kapital; stabilere Werte erhalten mehr Kapital.
- **Eignung:** Defensive Portfolios, die auf Stabilität und Diversifikation über Risikofaktoren hinweg setzen (ähnlich dem "All Weather" Ansatz).

### 5.3. Gleichgewichtung (Equal Weight - 1/N)
- **Prinzip:** Das Kapital wird gleichmäßig auf tất cả các mã verteilt: $w_i = \frac{1}{n}$.
- **Merkmale:** Maximale Diversifikation, keine Abhängigkeit von Schätzungen über zukünftige Renditen oder Volatilitäten.
- **Eignung:** Anleger, die Schätzfehler vermeiden wollen und an eine langfristige Outperformance durch maximale Streuung glauben.

### 5.4. Systembeschränkungen (System Constraints)
Um Realismus und Sicherheit zu gewährleisten, unterliegen tất cả các mã Modelle folgenden Regeln:
1.  **Vollinvestition (Full Investment):** $\sum w_i = 100\%$.
2.  **Konzentrationslimit (Concentration Cap):** Keine Aktie darf mehr als **40 %** ausmachen (für MVO/RP).
3.  **Mindestgewichtung (Min Weight Floor):** Der Benutzer kann eine Untergrenze festlegen (z. B. 2 %), um zu verhindern, dass der Optimierer eine bestehende Position vollständig auflöst.

---

## 6. Referenz-Prognosemodelle
Das System verwendet ein Ensemble von Deep-Learning-Architekturen:
- **LSTM (v7.2):** Optimiert für Zyklizität und zeitliche Stabilität.
- **Transformer (v8.0):** Optimiert für die Erkennung von Mustern bei hoher Volatilität.
- **PatchTST (v10.0):** Kanalunabhängige Verarbeitung, optimiert für fundamentaldatenbasierte Langfristprognosen.

---

## 7. Portfolio Performance & Risiko-Kennzahlen

Die Kennzahlen zur Bewertung der Portfolio-Gesundheit im Tab **Portfolio Builder**.

### 7.1. Weighted Return (Gewichtete Rendite)
Die tatsächliche Rendite des gesamten Portfolios basierend auf der Kapitalallokation:
$$R_p = \sum_{i=1}^{n} w_i R_i$$
Wobei $w_i$ das Gewicht und $R_i[t]$ die Rendite der Aktie $i$ ist.

### 7.2. Annual Vol (Annualisierte Volatilität)
Ein Maß für das systematische Risiko durch die Standardabweichung der Renditen:
$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$
Ein höherer Wert deutet auf stärkere Kursschwankungen hin.

### 7.3. Value at Risk (VaR 95%)
Der erwartete maximale Verlust an einem Tag mit einer Konfidenz von 95 %. Ein VaR von -2 % bedeutet, dass unter normalen Marktbedingungen eine 95 %ige Wahrscheinlichkeit besteht, dass das Portfolio an einem Tag nicht mehr als 2 % verliert.

### 7.4. Conditional Value at Risk (CVaR / Expected Shortfall)
Der durchschnittliche Verlust in den extremsten Szenarien (die restlichen 5 % außerhalb der VaR-Schwelle). CVaR beantwortet die Frage: *"Wie viel verliere ich im Durchschnitt, wenn ein extremer Markteinbruch eintritt?"*

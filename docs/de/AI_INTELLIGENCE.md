# 🧠 Unified Alpha-Risk Intelligence Hub (KI-Logik)

Dieses Dokument erläutert die Architektur und die Entscheidungslogik hinter dem integrierten KI-Analysesystem im Dashboard. Das System agiert als virtueller **Chief Investment Officer (CIO)** und gleicht quantitative Kennzahlen mit Marktnachrichten ab, um eine fundierte Investment-These zu erstellen.

## 1. Systemübersicht
Der **Unified Alpha-Risk Intelligence Hub** ist mehr als nur ein Datenzusammenfasser. Er führt eine "Signal-Konvergenz-Analyse" durch, indem er zwei unterschiedliche Welten analysiert:
1.  **Quantitativ:** Finanzkennzahlen, technische Indikatoren und algorithmische Bewertungen.
2.  **Qualitativ:** Nachrichten-Sentiment, regulatorische Risiken, makroökonomische Veränderungen und öffentliche Wahrnehmung.

## 2. Daten-Pipeline

### Quantitative Kennzahlen
Der KI werden umfassende "harte" Datenpunkte zur Verfügung gestellt:
- **AI Score (0-100):** Ein synthetischer Qualitätswert für die Fundamentaldaten.
- **Financial Momentum (FMI):** Echtzeit-Messung der Gewinn- und Umsatzbeschleunigung.
- **Technik:** RSI (Überkauft/Überverkauft), MA-Signale (Gleitende Durchschnittstrends).
- **Bewertung:** KGV (P/E), PEG Ratio, FCF-Marge (Free Cash Flow).
- **Marktumfeld:** Der globale Marktkontext (Bullish/Bearish/Neutral).

### Qualitative Intelligenz (NLP-Ergebnisse)
Über die Funktion `analyze_risk_with_llm` scannt das System 15 aktuelle Schlagzeilen von Google News:
- **Red Flag Score (0-100):** Bewertetes Risiko basierend auf dem Nachrichteninhalt.
- **Sentiment:** Grundstimmung (Positiv, Negativ, Kritisch).
- **Risikokategorie:** Schwerpunkt des Risikos (Rechtlich, Technisch, Finanziell, Reputational).

---

## 3. Signal-Alignment-Engine

Die KI analysiert die Beziehung zwischen diesen beiden Strömen, um potenzielle Konflikte zu erkennen:

- **KONVERGENZ (Alignment):** Wenn sowohl die Fundamentaldaten als auch das Nachrichten-Sentiment übereinstimmen (Bullish). Dies löst die höchsten Conviction-Ratings (Strong Buy) aus.
- **DIVERGENZ:** 
    - *Risiko-Konflikt:* Starke Fundamentaldaten, aber negative technische oder regulatorische Nachrichten. Die KI wird das Urteil oft herabstufen, um Kapital zu schützen.
    - *Chancen-Konflikt:* Schwache interne Daten, aber sehr positive Nachrichten (z. B. Gerüchte über eine Übernahme). Die KI identifiziert dies als spekulatives Risiko.
- **BEARISCHES ALIGNMENT:** Sowohl quantitative als auch qualitative Signale sind negativ. Die KI gibt ein Avoid/Reduce-Urteil ab.

---

## 4. Investment-Urteile (Aktionsvokabular)

Das System ist darauf beschränkt, eine von genau sechs definitiven Aktionen auszugeben:

| Aktion | Definition | Typische Bedingungen |
| :--- | :--- | :--- |
| **STRONG BUY** | Hohe Überzeugung | Perfekte Konvergenz, günstige Bewertung, starke Unterstützung durch Nachrichten. |
| **BUY** | Standardkauf | Gute Fundamentaldaten, keine nennenswerten negativen Nachrichten. |
| **WATCH & ACCUMULATE** | Taktische Akkumulation | Seitwärtsbewegung oder nachrichtenintensive Phasen mit Aufwärtspotenzial. |
| **HOLD** | Neutrale Position | Faire Bewertung ohne klaren unmittelbaren Katalysator. |
| **REDUCE** | Untergewicht | Erster Verschlechterung der Fundamentaldaten oder weniger negative Nachrichten. |
| **AVOID** | Verkaufen / Nicht Kaufen | Erhebliche Risiken erkannt (Red Flag > 70) oder starke fundamentale Verschlechterung. |

---

## 5. Operative Hinweise
- **CIO Persona:** Die KI ist mit einer kritischen Denkweise konzipiert. Sie kann mathematischen Formeln widersprechen, wenn sie qualitative Risiken wahrnimmt, die die Mathematik nicht erfassen kann.
- **Aktualisierungsfrequenz:** Nachrichten werden in Echtzeit gescannt, wenn die Taste gedrückt wird. Die Analyse ist für das aktuelle Handelsumfeld gültig.
- **API-Grenzwerte:** Das System nutzt derzeit den Cohere Trial-Tarif (begrenzt auf ca. 20 High-Fidelity-Aufrufe pro Monat).

> [!IMPORTANT]
> KI-Empfehlungen dienen nur zu Informationszwecken. Sie sind ein Werkzeug zur Entscheidungsunterstützung. Investoren sind für ihre eigenen finanziellen Entscheidungen selbst verantwortlich.


---

## 7. Smart Money Indikator v5.0

Der **Smart Money** Indikator verfolgt institutionelle Kauf- und Verkaufsmuster mithilfe der On-Balance Volume (OBV) Divergenzanalyse, um zu identifizieren, wo professionelle Investoren positioniert sind.

### Berechnungsmethodik (Erweitert v5.0)

**Zwei-Schichten-Architektur:**

**Schicht 1 - OBV-Divergenz (Priorität):**
- Erkennt, wenn OBV und Preis sich in ENTGEGENGESETZTE Richtungen bewegen
- **Hidden Accumulation (Versteckte Akkumulation)**: Preis fällt, aber OBV steigt → Institutionen kaufen heimlich
- **Hidden Distribution (Versteckte Distribution)**: Preis steigt, aber OBV fällt → Institutionen verkaufen in Rallyes
- Verwendet adaptives Fenster (15-25 Tage) basierend auf ATR/Volatilität
- Strengerer Magnitude-Filter (0,12 × avg_volume × window) zur Rauschfilterung

**Schicht 2 - OBV-Trend vs MA(21) (Fallback):**
- Klassischer institutioneller Fluss: OBV über/unter seinem 21-Tage-MA
- Erfordert 3 der letzten 5 Tage konsistent über/unter MA
- Wird nur angewendet, wenn keine klare Divergenz erkannt wird

**Hauptverbesserungen gegenüber v4.0:**
1. **Adaptives Fenster**: Hochvolatile Aktien verwenden breitere Fenster (25 Tage), niedrigvolatile verwenden schmalere (15 Tage)
2. **Strengerer Magnitude-Filter**: Erhöht von 0,05 auf 0,12 (240% avg volume Schwelle)
3. **Stärke-Bewertung**: Gibt Konfidenz-Score 0-100 zurück basierend auf:
   - OBV-Magnitude (40 Punkte)
   - Preis-Magnitude (25 Punkte)
   - Volumen-Bestätigung (20 Punkte)
   - Konsistenz über Fenster (15 Punkte)
4. **Layer-Erkennung**: Identifiziert, ob Signal von DIVERGENCE- oder TREND-Schicht kam

### Ausgabeformat

Gibt ein Dictionary mit drei Komponenten zurück:
```python
{
    "signal": "ACCUMULATION" | "DISTRIBUTION" | "NEUTRAL",
    "strength": 0-100,  # Konfidenz-Score
    "layer": "DIVERGENCE" | "TREND" | "NONE"
}
```

### Interpretation

| Signal | Stärke | Bedeutung | Aktion |
|---|---|---|---|
| **ACCUMULATION** | 70-100 | Starker institutioneller Kauf | Hohe Überzeugung Einstieg |
| **ACCUMULATION** | 40-69 | Moderater institutioneller Kauf | Vorsichtiger Einstieg |
| **ACCUMULATION** | 0-39 | Schwacher institutioneller Kauf | Nur beobachten, auf Bestätigung warten |
| **DISTRIBUTION** | 70-100 | Starker institutioneller Verkauf | Hohe Überzeugung Ausstieg |
| **DISTRIBUTION** | 40-69 | Moderater institutioneller Verkauf | Position reduzieren |
| **DISTRIBUTION** | 0-39 | Schwacher institutioneller Verkauf | Beobachten, Hedging erwägen |
| **NEUTRAL** | 0 | Kein klarer institutioneller Fluss | Auf klareres Signal warten |

**Layer-Priorität:**
- **DIVERGENCE**: Höchste Priorität (erfasst versteckte institutionelle Aktivität)
- **TREND**: Fallback (klassische OBV vs MA Bestätigung)

### Integration mit Strategien
- **Smart Money Accumulation** Strategie zielt auf ACCUMULATION-Signale mit strength ≥40
- **Distribution Warning** Strategie warnt vor DISTRIBUTION-Signalen mit strength ≥40
- **Oversold Reversal Setup** erfordert ACCUMULATION-Bestätigung mit strength ≥50

### Vorteile gegenüber traditionellen Methoden
- **Anpassung an Volatilität**: Fenstergröße passt sich automatisch an
- **Rauschfilterung**: Strengerer Magnitude-Filter reduziert Fehlsignale
- **Konfidenz-Bewertung**: Stärke-Metrik hilft, Signale zu priorisieren
- **Layer-Transparenz**: Wissen, ob Signal von Divergenz oder Trend kommt
- **Volumen-Bestätigung**: Aktuelle Volumenmuster validieren Signale

### Einschränkungen
- Basiert auf öffentlich verfügbaren Preis-/Volumendaten (kann Dark Pools nicht sehen)
- OBV ist kumulativ und pfadabhängig (verwendet letzte 126 Tage zur Vermeidung von Bias)
- Sollte mit anderen Indikatoren zur Bestätigung kombiniert werden
- Stärke-Bewertung ist relativ, keine absolute Wahrscheinlichkeit


---

## 8. 6-Säulen Institutionelles Bewertungssystem v14.0

Die **Institutional Rating Engine** synthetisiert sechs unabhängige Säulen, um umsetzbare Anlageempfehlungen zu generieren (STRONG BUY, BUY, HOLD, SELL, AVOID). Dieses System wird konsistent sowohl im Opportunity Radar Screener als auch im Deep Dive Tab verwendet.

### 8.1. Bewertungsarchitektur

**Funktion:** `compute_institutional_rating()` in `app.py`

**Säulen:**
1. **Technical Trend** (0-1 Punkte): MA-Signale, RSI-Bestätigung
2. **Quality** (0-1 Punkte): AI Score (fundamentale Qualität)
3. **Valuation** (0-1 Punkte): Sektorangepasstes P/E, PEG, Aufwärtspotenzial
4. **Risk** (0-1 Punkte): 52-Wochen-Position
5. **Conviction** (0-1 Punkte): Risk/Reward-Verhältnis
6. **Smart Money** (-1.25 bis +1.25 Punkte): Institutioneller Fluss mit stärkebasierter Bewertung

**Gesamtbereich:** -1.25 bis 6.25 Punkte

### 8.2. Smart Money Soft Scoring (NEU in v14.0)

Anstelle binärer 0/1-Punkte verwendet Smart Money jetzt **abgestufte Bewertung** basierend auf Signalstärke:

#### ACCUMULATION Bewertung

| Stärkebereich | Punkte | Label | Farbe |
|---|---|---|---|
| **≥ 80** | +1.25 | ACCUMULATION_STRONG | #00ffcc (Cyan) |
| **65-79** | +1.0 | ACCUMULATION_STRONG | #2ecc71 (Grün) |
| **40-64** | +0.5 | ACCUMULATION_WEAK | #3498db (Blau) |
| **< 40** | 0.0 | ACCUMULATION_WEAK | #95a5a6 (Grau) |

#### DISTRIBUTION Bewertung

| Stärkebereich | Punkte | Label | Farbe |
|---|---|---|---|
| **≥ 80** | -1.25 | DISTRIBUTION_STRONG | #c0392b (Dunkelrot) |
| **65-79** | -1.0 | DISTRIBUTION_STRONG | #e74c3c (Rot) |
| **40-64** | -0.5 | DISTRIBUTION_WEAK | #e67e22 (Orange) |
| **< 40** | 0.0 | DISTRIBUTION_WEAK | #95a5a6 (Grau) |

**Begründung:**
- Schwache Signale (< 40 Stärke) werden ignoriert, um Rauschen zu vermeiden
- Moderate Signale (40-64) erhalten halbes Gewicht
- Starke Signale (65-79) erhalten volles Gewicht
- Sehr starke Signale (≥ 80) erhalten Bonus-/Strafgewicht

### 8.3. Action Label Schwellenwerte

| Gesamtpunkte | Bedingungen | Action Label |
|---|---|---|
| **≥ 5.0** | Quality nicht schwach | **STRONG BUY** |
| **≥ 3.5** | Trend nicht bearish | **BUY / ACCUMULATE** |
| **≤ 2.0** | Trend + Valuation beide schwach | **SELL / AVOID** |
| **≤ 2.0** | Quality schwach | **SELL / AVOID** |
| **≤ 2.0** | Starke Distribution (SM ≤ -0.5) | **SELL / AVOID** |
| **≤ 2.5** | Quality stark | **HOLD / NEUTRAL** |
| **≤ 4.5** | RSI > 70 | **REDUCE / UNDERPERFORM** |
| **Andere** | - | **HOLD / NEUTRAL** |

### 8.4. Beispiele

#### Beispiel 1: Sehr Starker Accumulation Bonus
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ✅ (1.0)
R/R: ❌ (0.0)
Smart Money: ACCUMULATION (85, DIVERGENCE) → +1.25

Gesamt: 4.0 + 1.25 = 5.25 → STRONG BUY
```

#### Beispiel 2: Schwaches Signal Ignoriert
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ✅ (1.0)
R/R: ❌ (0.0)
Smart Money: ACCUMULATION (25, TREND) → +0.0

Gesamt: 4.0 + 0.0 = 4.0 → BUY (nicht STRONG BUY)
```

#### Beispiel 3: Distribution Strafe
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ❌ (0.0)
R/R: ❌ (0.0)
Smart Money: DISTRIBUTION (75, DIVERGENCE) → -1.0

Gesamt: 3.0 - 1.0 = 2.0 → SELL / AVOID
```

### 8.5. Vorteile des Soft Scoring

1. **Präzision:** Schwache OBV-Signale lösen kein STRONG BUY aus
2. **Qualität Belohnen:** Sehr starke Divergenz (≥80) erhält Bonusgewicht
3. **Risikomanagement:** Starke Distribution stuft Ratings aktiv herab
4. **Transparenz:** Benutzer sehen exakten Punktbeitrag
5. **Flexibilität:** Einfache Anpassung der Schwellenwerte ohne Code-Änderungen

### 8.6. Integration mit anderen Systemen

- **Opportunity Radar:** Verwendet Rating zum Filtern und Sortieren von Aktien
- **Deep Dive:** Zeigt 6-Säulen-Matrix mit Farbcodierung
- **AI Tab:** Integriert Rating in Konvergenzanalyse
- **Portfolio Builder:** Verwendet Rating für Positionsgrößenempfehlungen

---

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

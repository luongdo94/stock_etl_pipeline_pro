# 🏗️ ETL-Pipeline-Architektur

Dieses Dokument beschreibt die technische Architektur der Stock-ETL-Pipeline, von der Rohdatenerfassung bis zur Erstellung von "Analytics Ready"-Datensätzen im DuckDB Data Warehouse.

## 1. Design-Philosophie
Das System basiert auf drei Kernsäulen:
1.  **Euro-First (Normalisierung):** Alle Finanzkennzahlen (Kurs, Umsatz, Marktkapitalisierung) werden direkt an der Quelle in Euro umgerechnet, um eine absolute Vergleichbarkeit über globale Ticker hinweg zu gewährleisten.
2.  **Zero Down-time (Kontinuierliche Verfügbarkeit):** Verwendet einen Shadow-DB-Mechanismus und Atomic Swap, wodurch das Dashboard auch bei intensiven Datenladevorgängen funktionsfähig bleibt.
3.  **Data Layering (Daten-Schichtung):** Verwendet einen Modellierungsansatz im dbt-Stil (Raw -> Staging -> Intermediate -> Marts) für maximale Transparenz und Wartbarkeit.

---

## 2. Der 5-Stufen-Pipeline-Lebenszyklus

Das System wird über die Funktion `run_pipeline()` in `etl/pipeline.py` durch fünf strenge Phasen gesteuert:

### Schritt 0: Shadow-DB-Vorbereitung
Das System erstellt eine "Schattenkopie" (Shadow Copy) der Produktionsdatenbank. Alle neuen Schreibvorgänge werden auf dieser Kopie durchgeführt, um Auswirkungen auf Endbenutzer zu vermeiden, die gerade auf das Dashboard zugreifen.

### Schritt 1: Extraktion & Währungsnormalisierung
- **Quellen:** Yahoo Finance (yfinance) & Google News RSS.
- **Modi:** 
    - `INCREMENTAL`: Lädt nur Daten seit dem letzten Zeitstempel herunter (schnell, ~3-5 Sek.).
    - `FULL REFRESH`: Lädt das gesamte historische Fenster neu herunter (Standard: 5 Jahre).
- **Normalisierung:** Ruft automatisch Live-Wechselkurse ab (z. B. `USDEUR=X`), um Kurse und Fundamentaldaten während der Aufnahme direkt umzurechnen.

### Schritt 2: Validierung
Führt erste Integritätsprüfungen der extrahierten Daten durch (keine negativen Kurse, keine Nullwerte in kritischen Spalten). Wenn die Validierung fehlschlägt, wird der Prozess abgebrochen (Fail-fast).

### Schritt 3: Load (Laden)
Die Daten werden in das `raw`-Schema innerhalb von DuckDB geladen. Im inkrementellen Modus wird eine `UPSERT`-Strategie verwendet, um doppelte Datensätze zu vermeiden.

### Schritt 4: Transformation (Mehrschichtige Verarbeitung)
Dies ist die "Datenfabrik", in der SQL-Transformationen innerhalb von DuckDB stattfinden:
- **Staging Layer:** Bereinigung, Rundung und Kennzeichnung von Datensätzen.
- **Intermediate Layer:** Berechnung technischer Indikatoren (RSI, Gleitende Durchschnitte, Z-Score).
- **Marts Layer:** Endgültige Tabellen für die Analyse: Fact-Tables (Tägliche Renditen) und Dimension-Tables (Unternehmensinfo).

### Schritt 5: Atomic Swap
Sobald die Daten in der Shadow-DB bereit sind, führt das System einen physischen Dateiaustausch auf der Festplatte mitsamt einer Neuzuweisung der Verbindung durch. Dies geschieht in Millisekunden und stellt sicher, dass das Dashboard immer die neuesten Daten ohne Verbindungsfehler anzeigt.

---

## 3. Warehouse-Schema-Struktur

| Schema | Rolle | Typische Tabellen |
| :--- | :--- | :--- |
| **raw** | Unverarbeitete Rohdaten. | `stock_prices`, `company_info` |
| **staging** | Bereinigte Quelldaten. | `stg_stock_prices`, `stg_cashflows` |
| **intermediate** | Metrikberechnung (Business Logic). | `int_stock_metrics` (RSI, MA200...) |
| **marts** | Analytische Tabellen (BI Ready). | `dim_companies`, `fct_daily_returns` |

---

## 4. Qualitätskontrolle (Data Quality - DQ)

Am Ende jedes ETL-Zyklus führt das System eine automatisierte Reihe von Tests aus:
- **Kritische Tests:** Prüfungen auf Eindeutigkeit (Unique) und Pflichtfelder (Not Null). Ein Fehler führt zum Abbruch der Pipeline.
- **Soft-Tests:** Warnungen bei Lücken in den Fundamentaldaten. Die Ergebnisse werden in `marts.dq_warnings` geloggt und im Dashboard visualisiert.

---
> [!TIP]
> Sie können diesen Lebenszyklus über die Konsolen-Logs überwachen. Jede Phase wird zeitlich erfasst, um die Leistung zu optimieren und Transparenz zu schaffen.

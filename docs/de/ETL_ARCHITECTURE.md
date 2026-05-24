# 🏗️ ETL-Pipeline-Architektur

Dieses Dokument beschreibt die technische Architektur der Stock-ETL-Pipeline, von der Rohdatenerfassung bis zur Erstellung von "Analytics Ready"-Datensätzen im DuckDB Data Warehouse.

## 1. Design-Philosophie
Das System basiert auf drei Kernsäulen:
1.  **Euro-First (Normalisierung):** Alle Finanzkennzahlen (Kurs, Umsatz, Marktkapitalisierung) werden direkt an der Quelle in Euro umgerechnet, um eine absolute Vergleichbarkeit über globale Ticker hinweg zu gewährleisten.
2.  **Zero Down-time (Kontinuierliche Verfügbarkeit):** Verwendet einen Shadow-DB-Mechanismus und Atomic Swap, wodurch das Dashboard auch bei intensiven Datenladevorgängen funktionsfähig bleibt.
3.  **Data Layering (Daten-Schichtung):** Verwendet einen Modellierungsansatz im dbt-Stil (Raw -> Staging -> Intermediate -> Marts) für maximale Transparenz und Wartbarkeit.

---

## 2. Der 5-Stufen-Pipeline-Lebenszyklus

Das System wird über die Funktion `run_pipeline()` in `etl/pipeline.py` durch fünf strenge Phasen gesteuert, plus einem automatisierten Garbage-Collection-Schritt:

### Schritt 0: Shadow-DB-Vorbereitung
Das System erstellt eine "Schattenkopie" (Shadow Copy) der Produktionsdatenbank. Alle neuen Schreibvorgänge werden auf dieser Kopie durchgeführt, um Auswirkungen auf Endbenutzer zu vermeiden, die gerade auf das Dashboard zugreifen.

### Schritt 1: Extraktion & Währungsnormalisierung
- **Quellen:** Kombination aus `yahooquery` (Fundamentaldaten/Cashflows) und `yfinance` (Kursdaten/FX) für maximale Stabilität.
- **TradingView Auto-Discovery (NEU Mai 2026):** Dynamisches Ticker-Erkennungssystem, das die Abdeckung automatisch über die statische Konfiguration hinaus erweitert:
    - **5 Institutionelle Filter:** Value Stocks, GARP (Growth at Reasonable Price), Breakout Momentum, Quality Compounders, High-Yield Dividend.
    - **Globales Markt-Scanning:** Scannt 14 globale Märkte (USA, Europa, Asien-Pazifik) über die TradingView Scanner API.
    - **Intelligente Deduplizierung:** Verhindert doppelte Unternehmen (Cross-Listings, Vorzugsaktien, Hinterlegungsscheine) durch normalisierte Namensabgleichung.
    - **Börsen-Mapping:** Ordnet TradingView-Symbole automatisch Yahoo Finance-Tickern zu (z. B. `XETR:SIE` → `SIE.DE`).
    - **Top 20 pro Filter:** Ruft die Top 20 Aktien pro Filter ab, täglich dynamisch aktualisiert.
    - **Metadaten-Anreicherung:** Erfasst Sektor, Region und Erkennungsquelle für jeden automatisch erkannten Ticker.
- **Multi-Tier Smart Refresh Strategie:** Um die Geschwindigkeit zu maximieren und API-Drosselungen zu vermeiden, werden Daten in drei Frequenzen unterteilt:
    - **Tier 1 (Täglich - 24h):** Kursdaten und technische Indikatoren. Werden immer aktualisiert.
    - **Tier 2 (Taktisch - 7 Tage):** Quartalszahlen, Free Cash Flow (FCF) und Earnings-Kalender.
    - **Tier 3 (Strategisch - 30 Tage):** Unternehmens-Stammdaten (Sektoren, Industrien) und historische Jahresberichte.
- **Globale Marktabdeckung:** Alle Aktien werden unabhängig vom geografischen Standort gleich behandelt. Das System extrahiert Quartalsdaten für US-, europäische und asiatische Märkte ohne Diskriminierung (behoben Mai 2026 — zuvor wurden EU/Asien-Aktien fälschlicherweise gefiltert).
- **Normalisierung:** Ruft automatisch Live-Wechselkurse ab (z. B. `USDEUR=X`), um alle Werte bei der Aufnahme direkt in Euro zu normalisieren.

### Schritt 2: Validierung
Führt erste Integritätsprüfungen der extrahierten Daten durch (keine negativen Kurse, keine Nullwerte in kritischen Spalten). Wenn die Validierung fehlschlägt, wird der Prozess abgebrochen (Fail-fast).

### Schritt 3: Load (Laden)
Die Daten werden in das `raw`-Schema innerhalb von DuckDB geladen. Im inkrementellen Modus wird eine `UPSERT`-Strategie verwendet, um doppelte Datensätze zu vermeiden.

### Schritt 4: Transformation (Mehrschichtige Verarbeitung)
Dies ist die "Datenfabrik", in der SQL-Transformationen innerhalb von DuckDB stattfinden:
- **Aktive Ticker-Filterung (NEU Mai 2026):** Die Staging-Schicht filtert jetzt "tote" oder veraltete Ticker heraus, die nicht mehr im aktiven Ticker-Pool sind, um zu verhindern, dass Zombie-Daten die Marts verschmutzen.
- **Staging Layer:** Bereinigung, Rundung und Kennzeichnung von Datensätzen.
- **Intermediate Layer:** Berechnung technischer Indikatoren (RSI, Gleitende Durchschnitte, Z-Score).
- **Marts Layer:** Endgültige Tabellen für die Analyse: Fact-Tables (Tägliche Renditen) und Dimension-Tables (Unternehmensinfo).

### Schritt 4.8: Garbage Collection (NEU Mai 2026)
Automatisiertes Bereinigungssystem zur Aufrechterhaltung der Data-Warehouse-Hygiene:
- **Entfernung veralteter Ticker:** Automatisch erkannte TradingView-Ticker, die seit 7+ Tagen nicht aktualisiert wurden, werden automatisch aus allen Raw-Tabellen gelöscht.
- **Schutz der Basis-Ticker:** In `config/tickers.yaml` definierte Ticker sind streng geschützt und werden niemals entfernt.
- **Begründung:** TradingView-Filter geben "Top 20"-Rankings zurück, die sich täglich ändern. Aktien, die aus den Rankings fallen, werden veraltet und sollten entfernt werden, um eine Aufblähung der Datenbank zu verhindern.
- **Umfang:** Löscht aus 9 Raw-Tabellen: `stock_prices`, `company_info`, `historical_financials`, `quarterly_financials`, `cashflows`, `earnings_calendar`, `earnings_surprise`, `forward_estimates`, `hist_fcf`, `hist_fcf_quarterly`.
- **Sicherheit:** Läuft nur nach erfolgreicher Transformation, niemals bei Basis-Konfigurations-Tickern.

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

---

## 5. TradingView Auto-Discovery System (NEU Mai 2026)

Die Pipeline verfügt nun über ein intelligentes dynamisches Ticker-Erkennungssystem, das die Abdeckung automatisch über die statische Konfiguration hinaus erweitert.

### 5.1. Architektur-Übersicht

**Traditioneller Ansatz (Vor Mai 2026):**
- Statische Ticker-Liste in `config/tickers.yaml`
- Manuelle Updates erforderlich, um neue Aktien hinzuzufügen
- Begrenzt auf ~600 vorkonfigurierte Ticker

**Auto-Discovery-Ansatz (Mai 2026+):**
- Dynamische Ticker-Erkennung über TradingView Scanner API
- Automatische Erweiterung auf 700+ Ticker weltweit
- Tägliche Aktualisierung der Top-Performance-Aktien pro Filter

### 5.2. Fünf institutionelle Filter

| Filter | Kriterien | Zielprofil |
|---|---|---|
| **Value Stocks** | KGV < 15, KBV < 1,5, Dividendenrendite > 2% | Unterbewertete Dividendenzahler |
| **GARP** | EPS-Wachstum > 15%, Umsatzwachstum > 10%, KGV < 25 | Wachstum zu angemessenem Preis |
| **Breakout Momentum** | Kurs > MA50 > MA200, RSI 60-75, Volumen > 1M | Technische Ausbrüche |
| **Quality Compounders** | ROIC > 15%, ROE > 20%, Betriebsmarge > 15%, Verschuldungsgrad < 0,5 | Hochwertige Unternehmen |
| **High-Yield Dividend** | Dividendenrendite > 4%, Ausschüttungsquote < 60%, Umsatzwachstum > 0% | Nachhaltiges Einkommen |

### 5.3. Globale Marktabdeckung

Scannt 14 Märkte: `america`, `vietnam`, `uk`, `germany`, `france`, `japan`, `hongkong`, `china`, `australia`, `canada`, `india`, `brazil`, `taiwan`, `korea`

### 5.4. Börsen-Mapping-Logik

Konvertiert TradingView-Symbole automatisch in Yahoo Finance-Ticker:

```python
XETR:SIE    → SIE.DE     (Frankfurt)
LSE:BP      → BP.L       (London)
HOSE:VNM    → VNM.VN     (Vietnam)
TSE:7203    → 7203.T     (Tokio)
NASDAQ:AAPL → AAPL       (USA)
```

### 5.5. Intelligente Deduplizierung

Verhindert doppelte Unternehmen durch normalisierte Namensabgleichung:

1. **Unternehmensnamen normalisieren:** Suffixe entfernen (Inc, Corp, Ltd), Sonderzeichen, zusätzliche Leerzeichen
2. **Cross-Listing-Erkennung:** Überspringen, wenn normalisierter Name mit vorhandenem Ticker übereinstimmt
3. **Vorzugsaktien-Filterung:** Ticker mit "Preferred", "PFD", "Depositary Share", "Warrant" im Namen ausschließen
4. **Ticker-Validierung:** Ticker mit Leerzeichen, Schrägstrichen oder ungültigen Zeichen überspringen

**Beispiel:**
```
Basis-Konfiguration: AAPL (Apple Inc.)
TradingView gibt zurück: AAPL (Apple Inc.), AAPL34 (Apple BDR Brasilien)
Ergebnis: Nur AAPL behalten (AAPL34 als Duplikat gefiltert)
```

### 5.6. Integration in die ETL-Pipeline

**Funktionsablauf:**
```python
# etl/extract.py
base_tickers = load_tickers_config()           # Lade config/tickers.yaml
dynamic_tickers = fetch_dynamic_tv_tickers()   # Abrufen von TradingView
TICKERS = {**dynamic_tickers, **base_tickers} # Zusammenführen (Basis hat Vorrang)
```

**Pipeline-Integration:**
- `etl/pipeline.py`: Übergibt kombinierte `TICKERS` an Smart Recovery und Transform-Phasen
- `etl/transform.py`: Filtert Staging-Views, um nur aktive Ticker einzuschließen
- `etl/load.py`: Garbage Collection entfernt veraltete automatisch erkannte Ticker

### 5.7. Lebenszyklus-Management

**Tag 1-7:** Automatisch erkannter Ticker wird aktiv verfolgt
- Erscheint in TradingView-Filterergebnissen
- Daten werden täglich extrahiert und geladen
- Im Dashboard mit "TV_"-Erkennungsquellen-Tag sichtbar

**Tag 8+:** Ticker fällt aus den Top-20-Rankings
- Wird nicht mehr von der TradingView-API zurückgegeben
- `_extracted_at`-Zeitstempel wird veraltet (>7 Tage alt)
- Garbage Collection entfernt aus allen Raw-Tabellen
- Verschwindet aus dem Dashboard

**Wiedererkennung:** Wenn Ticker wieder in die Top 20 eintritt, wird er automatisch wieder hinzugefügt

### 5.8. Dashboard-Integration

**Auto-Discovery-Bereich (Übersichts-Tab):**
- Zeigt Anzahl neu entdeckter Aktien
- Listet Ticker mit Erkennungsquellen-Tags auf
- Unterscheidet von Basis-Konfigurations-Aktien

**TradingView-Filter-Tab (Screener):**
- Echtzeit-Filterergebnisse nach Quelle gruppiert
- Aktienkarten mit Sektor-/Regions-Metadaten
- Status-Badges (Vorhandene DB vs. Neue Entdeckung)
- Zusammenfassende Metriken (Gesamtsignale, aktive Filter, Top-Performer)

### 5.9. Leistungsüberlegungen

- **API-Ratenlimits:** 5 Filter × 20 Aktien = 100 API-Aufrufe pro Durchlauf (weit innerhalb der Grenzen)
- **Deduplizierungs-Overhead:** O(n) normalisierter Namensvergleich (vernachlässigbar für <1000 Ticker)
- **Speicher-Auswirkung:** ~100 zusätzliche Ticker × 9 Tabellen = minimal (DuckDB handhabt effizient)
- **Garbage Collection:** Läuft in <1 Sekunde (einfaches DELETE mit Datumsfilter)

---

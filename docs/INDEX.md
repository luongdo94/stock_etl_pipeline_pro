# 📚 Documentation Index

Complete documentation for the Stock ETL Pipeline & Analytics Platform.

---

## 🗂️ Documentation Structure

### **Core Documentation (English)**

#### **Algorithms & Analytics**
- **[en/ALGORITHMS.md](./en/ALGORITHMS.md)** — Quantitative algorithms
- **[en/DECISION_RULES.md](./en/DECISION_RULES.md)** — How a stance is reached: BUY gates, blocks, valuation reliability
  - Quality Score v4.1
  - FMI (Fundamental Momentum Index)
  - **Smart Money Detection v6.0** (Section 3) — NEW
  - Trend Confidence & Market Regime
  - Zone-Based Support & Resistance v2.0
  - Portfolio Optimization (Max Sharpe, Risk Parity, Equal Weight)
  - AI Trading Signature
  - Performance Metrics (Sharpe, Sortino, VaR, CVaR)

#### **AI & Intelligence**
- **[en/AI_INTELLIGENCE.md](./en/AI_INTELLIGENCE.md)** — AI systems
  - LLM analysis (Cohere Command-R+)
  - News sentiment (FinBERT)
  - Risk scoring
  - Convergence analysis
  - Investment verdicts

#### **ETL Architecture**
- **[en/ETL_ARCHITECTURE.md](./en/ETL_ARCHITECTURE.md)** — Pipeline design
  - Pipeline stages (Extract, Validate, Load, Transform, Atomic Swap)
  - Shadow DB pattern
  - Multi-tier refresh strategy
  - Schema structure (raw/staging/intermediate/marts)

#### **Data Quality**
- **[en/DATA_QUALITY.md](./en/DATA_QUALITY.md)** — DQ framework
  - DQ rules & validation layers
  - Multi-pass extraction
  - Coverage thresholds
  - Error handling

#### **Testing**
- **[en/TESTING.md](./en/TESTING.md)** — Test framework
  - Test structure
  - pytest configuration
  - Mocking strategies
  - CI/CD integration

#### **API Documentation**
- **[en/API.md](./en/API.md)** — API reference
  - Endpoints
  - Authentication
  - Request/Response formats

---

### **Translations**

#### **Vietnamese (Tiếng Việt)**
- **[vi/ALGORITHMS.md](./vi/ALGORITHMS.md)** — Thuật toán
- **[vi/AI_INTELLIGENCE.md](./vi/AI_INTELLIGENCE.md)** — Hệ thống AI
- **[vi/ETL_ARCHITECTURE.md](./vi/ETL_ARCHITECTURE.md)** — Kiến trúc ETL
- **[vi/DATA_QUALITY.md](./vi/DATA_QUALITY.md)** — Chất lượng dữ liệu
- **[vi/TESTING.md](./vi/TESTING.md)** — Kiểm thử
- **[vi/SYSTEM_GUIDE.md](./vi/SYSTEM_GUIDE.md)** — Hướng dẫn hệ thống

#### **German (Deutsch)**
- **[de/ALGORITHMS.md](./de/ALGORITHMS.md)** — Algorithmen
- **[de/AI_INTELLIGENCE.md](./de/AI_INTELLIGENCE.md)** — KI-Systeme
- **[de/ETL_ARCHITECTURE.md](./de/ETL_ARCHITECTURE.md)** — ETL-Architektur

---

### **Status & Release Notes**

- **[status/CURRENT_STATUS.md](archive/status/CURRENT_STATUS.md)** — Current system status
- **[status/SMART_MONEY_V6_RELEASE.md](archive/status/SMART_MONEY_V6_RELEASE.md)** — Smart Money v6.0 release notes
- **[status/RATING_SYSTEM_V14_UPGRADE.md](archive/status/RATING_SYSTEM_V14_UPGRADE.md)** — Rating system v14
- **[status/SMART_MONEY_V5_UPGRADE.md](archive/status/SMART_MONEY_V5_UPGRADE.md)** — Smart Money v5 (legacy)
- **[status/IMPLEMENTATION_COMPLETE.md](archive/status/IMPLEMENTATION_COMPLETE.md)** — Implementation status
- **[status/DATA_FLOW_EXPLANATION.md](archive/status/DATA_FLOW_EXPLANATION.md)** — Data flow
- **[status/SCANNER_STRATEGIES_UPDATE.md](archive/status/SCANNER_STRATEGIES_UPDATE.md)** — Scanner strategies
- **[status/MERGE_STATUS_REPORT.md](archive/status/MERGE_STATUS_REPORT.md)** — Merge status
- **[status/GIT_PUSH_SUCCESS.md](archive/status/GIT_PUSH_SUCCESS.md)** — Git push status

---

### **Smart Money v6.0 Documentation**

Complete documentation for Smart Money Detection v6.0 upgrade:

**[archive/smart-money-v6/](./archive/smart-money-v6/)** — Dedicated folder
- **[README.md](archive/smart-money-v6/README.md)** — Documentation index
- **[README_SMART_MONEY_V6.md](archive/smart-money-v6/README_SMART_MONEY_V6.md)** — Quick start guide
- **[UPGRADE_SUMMARY.md](archive/smart-money-v6/UPGRADE_SUMMARY.md)** — Quick reference
- **[VISUAL_SUMMARY.md](archive/smart-money-v6/VISUAL_SUMMARY.md)** — Visual overview
- **[SMART_MONEY_V6_UPGRADE.md](archive/smart-money-v6/SMART_MONEY_V6_UPGRADE.md)** — Technical details
- **[IMPLEMENTATION_CHECKLIST.md](archive/smart-money-v6/IMPLEMENTATION_CHECKLIST.md)** — Progress tracking
- **[SKILL_UPDATE_SUMMARY.md](archive/smart-money-v6/SKILL_UPDATE_SUMMARY.md)** — Skills update
- **[FINAL_SUMMARY.md](archive/smart-money-v6/FINAL_SUMMARY.md)** — Complete overview

---

### **Data Dictionaries**

- **[data_dictionary/en.md](./data_dictionary/en.md)** — English
- **[data_dictionary/vi.md](./data_dictionary/vi.md)** — Vietnamese
- **[data_dictionary/fr.md](./data_dictionary/fr.md)** — French

---

### **Architecture & Design**

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — System architecture overview
- **[README.md](./README.md)** — Documentation overview

---

## 🎯 Quick Navigation

### **By Role:**

#### **Traders/Users:**
- Start: [archive/smart-money-v6/README_SMART_MONEY_V6.md](archive/smart-money-v6/README_SMART_MONEY_V6.md)
- Algorithms: [en/ALGORITHMS.md](./en/ALGORITHMS.md)
- AI Systems: [en/AI_INTELLIGENCE.md](./en/AI_INTELLIGENCE.md)
- Vietnamese: [vi/SYSTEM_GUIDE.md](./vi/SYSTEM_GUIDE.md)

#### **Developers:**
- Architecture: [en/ETL_ARCHITECTURE.md](./en/ETL_ARCHITECTURE.md)
- Testing: [en/TESTING.md](./en/TESTING.md)
- API: [en/API.md](./en/API.md)
- Smart Money v6: [archive/smart-money-v6/SMART_MONEY_V6_UPGRADE.md](archive/smart-money-v6/SMART_MONEY_V6_UPGRADE.md)

#### **Data Engineers:**
- ETL: [en/ETL_ARCHITECTURE.md](./en/ETL_ARCHITECTURE.md)
- Data Quality: [en/DATA_QUALITY.md](./en/DATA_QUALITY.md)
- Data Dictionary: [data_dictionary/en.md](./data_dictionary/en.md)

#### **Project Managers:**
- Status: [status/CURRENT_STATUS.md](archive/status/CURRENT_STATUS.md)
- Releases: [status/](./archive/status/)
- Smart Money v6: [archive/smart-money-v6/FINAL_SUMMARY.md](archive/smart-money-v6/FINAL_SUMMARY.md)

#### **AI Agents:**
- Skills: [../.agents/skills/](../.agents/skills/)
- Algorithms: [en/ALGORITHMS.md](./en/ALGORITHMS.md)
- Smart Money: [archive/smart-money-v6/SMART_MONEY_V6_UPGRADE.md](archive/smart-money-v6/SMART_MONEY_V6_UPGRADE.md)

---

## 🆕 Recent Updates

### **May 2, 2026:**
- ✅ **Smart Money Detection v6.0** — Major upgrade
  - Added MFI cross-validation
  - Added Layer 2 (Institutional Volume Pattern)
  - Added Volume Quality Scoring
  - Added sector-specific thresholds
  - -36% false positives, +10% true positives
  - Documentation: [archive/smart-money-v6/](./archive/smart-money-v6/)

---

## 📊 Documentation Statistics

- **Total Files:** 30+
- **Languages:** 3 (English, Vietnamese, German)
- **Core Docs:** 5 (ALGORITHMS, AI_INTELLIGENCE, ETL_ARCHITECTURE, DATA_QUALITY, TESTING)
- **Status Reports:** 9
- **Smart Money v6.0 Docs:** 8
- **Data Dictionaries:** 3

---

## 🔗 External Resources

- **GitHub Repository:** [Link to repo]
- **Live Dashboard:** [Link to dashboard]
- **API Endpoint:** [Link to API]
- **Support:** [Contact info]

---

**Last Updated:** May 2, 2026  
**Version:** Platform v2.5 | Smart Money v6.0  
**Status:** ✅ Production Ready

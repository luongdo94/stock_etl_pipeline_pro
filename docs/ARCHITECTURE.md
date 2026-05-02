# Honest Quant Intelligence Platform - Knowledge Graph / Architecture

Dưới đây là sơ đồ tri thức (Knowledge Graph) tổng quan về kiến trúc hệ thống, phản ánh chi tiết các Engine AI và Data Pipeline.

```mermaid
graph TD
    %% Styling
    classDef source fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    classDef etl fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff;
    classDef db fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
    classDef ai fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff;
    classDef ui fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff;
    classDef orchestrator fill:#c0392b,stroke:#e74c3c,stroke-width:2px,color:#fff;

    %% Orchestration
    Airflow(("Apache Airflow<br/>(Docker Orchestration)")):::orchestrator -.-> |Triggers Daily| Extract

    %% Data Sources
    subgraph group1 [1. External Data Sources]
        YF["yfinance API<br/>(Price, Volume, FX)"]:::source
        YQ["yahooquery API<br/>(Financials, Earnings)"]:::source
    end

    %% ETL Layer
    subgraph group2 [2. Data Engineering Pipeline ETL]
        Extract["etl/extract.py<br/>Multi-Pass & Smart Recovery"]:::etl
        Transform["etl/transform.py<br/>Normalization & Modeling"]:::etl
        Quant["etl/utils.py<br/>Quant Diagnostics Engine<br/>(Z-Score, FMI, Quality)"]:::etl
        DQ["etl/dq_engine.py<br/>Data Quality Guardrails"]:::etl
        Load["etl/load.py<br/>Incremental Loading"]:::etl
        
        Extract --> Transform
        Transform --> Quant
        Quant --> DQ
        DQ --> Load
    end

    %% Links from Sources to ETL
    YF --> Extract
    YQ --> Extract

    %% Data Warehouse
    subgraph group3 [3. Data Warehouse & Storage]
        RAW[("Raw Data<br/>JSON Dumps")]:::db
        DIM[("marts.dim_companies<br/>Company Metadata")]:::db
        FCT[("marts.fct_daily_returns<br/>Prices & Tech Indicators")]:::db
        FIN[("marts.dim_financials<br/>Quarterly & Annual Data")]:::db
        Supa[("Supabase (Cloud)<br/>Watchlist & Portfolio")]:::db
        
        Load --> RAW
        Load --> DIM
        Load --> FCT
        Load --> FIN
    end

    %% AI / ML Engine
    subgraph group4 [4. AI Predictive Suite]
        LSTM["LSTM Core<br/>(Mean-Reversion)"]:::ai
        Trans["Transformer<br/>(High-Vol Pattern Recognition)"]:::ai
        Patch["PatchTST (SOTA)<br/>(Channel-Independent)"]:::ai
        MC["Monte Carlo Simulation<br/>(Value-at-Risk)"]:::ai
        CIO["LLM CIO Agent<br/>(Portfolio Review)"]:::ai
        
        FCT --> LSTM
        FCT --> Trans
        FCT --> Patch
        FCT --> MC
        DIM --> CIO
        FCT --> CIO
    end

    %% UI Dashboard
    subgraph group5 [5. Tactical Dashboard Streamlit]
        T1["Macro Overview<br/>Market Breadth"]:::ui
        T2["Single Stock Deep Dive<br/>Radar Charts"]:::ui
        T3["Market Scanner<br/>Opportunity Radar"]:::ui
        T4["Strategy Backtester V2<br/>Trade Simulation"]:::ui
        T7["Portfolio Builder<br/>Rebalancing & Optimization"]:::ui
        
        DIM --> T1
        FCT --> T1
        
        DIM --> T2
        FCT --> T2
        FIN --> T2
        
        DIM --> T3
        FCT --> T3
        FIN --> T3
        
        FCT --> T4
        
        DIM --> T7
        FCT --> T7
        FIN --> T7
        
        %% Supabase logic
        T7 <--> Supa
        T1 <--> Supa
        
        LSTM -.-> T2
        Trans -.-> T2
        Patch -.-> T2
        MC -.-> T7
        CIO -.-> T7
    end
```

### Các cập nhật và thành phần cốt lõi:

1. **AI Predictive Suite (Bộ 3 Mô Hình Forecasting):**
   - **LSTM Core:** Dùng để dự đoán Mean-Reversion.
   - **Transformer:** Nhận diện mẫu đồ thị (Pattern Recognition) trong giai đoạn biến động cao (High-Vol).
   - **PatchTST (SOTA):** Mô hình tối tân phân tích độc lập theo chuỗi thời gian (Channel-Independent) để nhận diện các thay đổi cơ bản.
2. **Data Quality Engine (DQ Engine):** `etl/dq_engine.py` đóng vai trò rào chắn dữ liệu quan trọng giữa khâu tính toán Quant Metrics và khâu ghi vào Data Warehouse.
3. **Database Kép (DuckDB & Supabase):**
   - **DuckDB** lưu trữ toàn bộ dữ liệu thị trường.
   - **Supabase Cloud** được sử dụng để quản lý state của người dùng (Portfolio và Watchlist), đồng bộ hai chiều trực tiếp với Dashboard.
4. **Data Sources & Orchestration**: Điều khiển bởi Airflow, lấy dữ liệu từ `yfinance` và `yahooquery` qua cơ chế *Multi-Pass Extract*.

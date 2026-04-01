---
name: data-engineering
description: >
  Hỗ trợ xây dựng data pipeline với Python, Airflow, DuckDB, SQL.
  Dùng khi tạo ETL/ELT pipeline, viết DAG, thiết kế schema,
  tối ưu query, hoặc xử lý dữ liệu lớn.
---

# Data Engineering Skill

## Khi nào dùng skill này
- Tạo hoặc debug Airflow DAG
- Viết query DuckDB / SQL Server phức tạp
- Thiết kế schema (star schema, data vault)
- Xây ETL/ELT pipeline đọc từ CSV/API/ERP
- Data quality checks và monitoring

## Kiến trúc pipeline chuẩn
Source (CSV/API/DB)
→ Extract (Python / Airbyte)
→ Validate (Great Expectations / custom checks)
→ Transform (dbt / DuckDB SQL)
→ Load (DuckDB / Postgres / SQL Server)
→ Reporting (SSRS / Power BI)

## DuckDB best practices
- Dùng `read_csv_auto()` và `read_parquet()` cho file-based ingestion
- ASOF JOIN cho time series lookups
- Window functions thay vì self-join
- `COPY TO` cho export parquet/csv nhanh

## Airflow best practices
- TaskFlow API cho pipeline Python-heavy
- Sensors cho external dependency
- Dynamic task mapping cho multi-ticker
- SLAs + Alerts cho pipeline critical


"""Screener rows: what the user is told when a source could not be read (G4-05)."""
import duckdb
import pandas as pd


def test_a_quarterly_read_error_warns_real_rows_without_changing_decisions(tmp_path):
    from core.screener import build_screener_table
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    clean = build_screener_table(frames[1], frames[0], frames[4], frames[3], frames[7])
    assert not clean.empty and clean[["Price", "Quality"]].notna().all().all()
    assert all("could not be read" not in w for warnings in clean["_warnings"] for w in warnings)

    frames[4].attrs["error"] = "IOException: boom"
    failed = build_screener_table(frames[1], frames[0], frames[4], frames[3], frames[7])
    assert all(any("could not be read" in w and "boom" in w for w in warnings) for warnings in failed["_warnings"])
    pd.testing.assert_frame_equal(failed[["Decision", "Action"]], clean[["Decision", "Action"]])
    pd.testing.assert_frame_equal(failed.select_dtypes(include="number"), clean.select_dtypes(include="number"))

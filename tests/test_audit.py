import unittest
import os
import uuid
import duckdb
import pandas as pd
from pathlib import Path
from etl.pipeline import AuditManager

# Mocking AUDIT_DB_PATH for testing
TEST_DB_PATH = "warehouse/test_audit.duckdb"

class TestAuditManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare a fresh test database."""
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        
        # Ensure the warehouse dir exists
        Path("warehouse").mkdir(exist_ok=True)
        
        # Patch the AUDIT_DB_PATH used in AuditManager for this session
        import etl.pipeline
        cls.original_db_path = etl.pipeline.AUDIT_DB_PATH
        etl.pipeline.AUDIT_DB_PATH = TEST_DB_PATH

    @classmethod
    def tearDownClass(cls):
        """Restore original AUDIT_DB_PATH and cleanup."""
        import etl.pipeline
        etl.pipeline.AUDIT_DB_PATH = cls.original_db_path
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_audit_lifecycle_success(self):
        """Verify that a successful run is correctly logged."""
        mode = "TEST_SUCCESS"
        with AuditManager(mode=mode) as audit:
            audit.rows_processed = 100
            run_id = audit.run_id
        
        # Verify in DB
        with duckdb.connect(TEST_DB_PATH) as conn:
            res = conn.execute("SELECT status, rows_processed, mode FROM etl.audit_log WHERE run_id = ?", [run_id]).fetchone()
            self.assertIsNotNone(res)
            self.assertEqual(res[0], "SUCCESS")
            self.assertEqual(res[1], 100)
            self.assertEqual(res[2], mode)

    def test_audit_lifecycle_failure(self):
        """Verify that a failed run captures the error traceback."""
        mode = "TEST_FAILURE"
        run_id = None
        
        try:
            with AuditManager(mode=mode) as audit:
                run_id = audit.run_id
                raise ValueError("Simulated Pipeline Crash")
        except ValueError:
            pass # Expected
            
        # Verify in DB
        with duckdb.connect(TEST_DB_PATH) as conn:
            res = conn.execute("SELECT status, error_message FROM etl.audit_log WHERE run_id = ?", [run_id]).fetchone()
            self.assertIsNotNone(res)
            self.assertEqual(res[0], "FAILED")
            self.assertIn("Simulated Pipeline Crash", res[1])
            self.assertIn("traceback", res[1].lower())

    def test_audit_incremental_updates(self):
        """Verify that we can update rows_processed multiple times."""
        with AuditManager(mode="INCREMENTAL_TEST") as audit:
            run_id = audit.run_id
            audit.rows_processed += 50
            audit._log_to_db() # Manual intermediate sync
            
            audit.rows_processed += 25
            # Exit will auto-sync
            
        with duckdb.connect(TEST_DB_PATH) as conn:
            val = conn.execute("SELECT rows_processed FROM etl.audit_log WHERE run_id = ?", [run_id]).fetchone()[0]
            self.assertEqual(val, 75)

if __name__ == "__main__":
    unittest.main()

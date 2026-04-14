# scratch/patch_transform.py
import os

FILE_PATH = "etl/transform.py"

def patch():
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, "r") as f:
        content = f.read()

    # 1. Patch Quarterly (Use _loaded_at as extraction time is missing)
    old_q = """        FROM raw.quarterly_financials
        ORDER BY ticker, date"""
    new_q = """        FROM raw.quarterly_financials
        -- DEDUPLICATION: Take the most recent extraction for each ticker/date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, date ORDER BY _loaded_at DESC) = 1
        ORDER BY ticker, date"""
    
    # 2. Patch Annual (Use _loaded_at)
    old_a = """        FROM raw.historical_financials
        ORDER BY ticker, year"""
    new_a = """        FROM raw.historical_financials
        -- DEDUPLICATION: Take the most recent extraction for each ticker/date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, date ORDER BY _loaded_at DESC) = 1
        ORDER BY ticker, year"""

    # 3. Patch the redundant dim_quarterly/annual calls later in the file if they exist
    # (Actually they are lines 270 and 282)
    
    patched_content = content.replace(old_q, new_q).replace(old_a, new_a)

    if patched_content == content:
        print("⚠️ No changes made. Patterns might not match exactly.")
        return

    with open(FILE_PATH, "w") as f:
        f.write(patched_content)
    print("✅ Successfully patched etl/transform.py with CORRECT timestamp columns!")

if __name__ == "__main__":
    patch()

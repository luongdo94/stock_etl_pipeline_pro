# scratch/master_patch_transform.py
import os

FILE_PATH = "etl/transform.py"

def master_patch():
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, "r") as f:
        lines = f.readlines()

    new_lines = []
    skip_mode = False
    
    # We want to keep the definitions inside Step 0 (lines ~250-285)
    # and remove the redundant ones later (lines ~505-550)
    
    for i, line in enumerate(lines):
        # Detect the start of the redundant blocks
        if "DIMENSION: Historical Annual Financials" in line and i > 400:
            print(f"✂️ Removing redundant Annual Financials at line {i+1}")
            skip_mode = True
        elif "DIMENSION: Historical Quarterly Financials" in line and i > 400:
            print(f"✂️ Removing redundant Quarterly Financials at line {i+1}")
            skip_mode = True
        elif "logger.info(\"✅ Mart tables created" in line and skip_mode:
            # Re-enable keeping lines at the very end
            skip_mode = False
            new_lines.append(line)
        elif not skip_mode:
            # Also fix any remaining _extracted_at for financials in the kept sections
            if "raw.historical_financials" in lines[max(0, i-5):i+5] or "raw.quarterly_financials" in lines[max(0, i-5):i+5]:
                line = line.replace("_extracted_at", "_loaded_at")
            new_lines.append(line)

    with open(FILE_PATH, "w") as f:
        f.writelines(new_lines)
    print("✅ Successfully cleaned up redundancies and fixed columns in etl/transform.py!")

if __name__ == "__main__":
    master_patch()

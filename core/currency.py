"""core/currency.py — the ONE definition of currency units (used by the ETL and by valuation)."""
from typing import Optional

# Minor units Yahoo uses for QUOTES (prices, targets). Company-level amounts are reported in the MAJOR unit.
MINOR_UNITS = {"GBp": "GBP", "GBX": "GBP", "GBx": "GBP", "ZAc": "ZAR", "ILA": "ILS"}


def major_currency(ccy) -> Optional[str]:
    if ccy is None:
        return None
    c = str(ccy).strip()
    if c in ("", "nan", "None"):
        return None
    return MINOR_UNITS.get(c, c.upper())

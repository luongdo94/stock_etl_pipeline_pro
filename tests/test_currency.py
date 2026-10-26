"""Currency aliases, minor units and missing currency values."""
import pandas as pd
import pytest

from core import currency


@pytest.mark.parametrize("code,minor,major", [(" GBp ", True, "GBP"), ("GBp", True, "GBP"), (" ZAc", True, "ZAR"),
                                               ("GBP", False, "GBP"), (None, False, None),
                                               (pd.NA, False, None), (pd.NaT, False, None), (float("nan"), False, None)])
def test_minor_unit_detection_strips_spaces(code, minor, major):
    # K7-04, A1-06: padded minor units and missing currency objects.
    assert currency.is_minor(code) is minor
    assert currency.major_currency(code) == major

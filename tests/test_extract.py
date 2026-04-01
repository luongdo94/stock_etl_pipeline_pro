# tests/test_extract.py
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.extract import _guess_currency


class TestGuessCurrency:
    """Test currency detection from ticker symbols."""
    
    def test_us_ticker(self):
        assert _guess_currency("AAPL") == "USD"
        assert _guess_currency("MSFT") == "USD"
        assert _guess_currency("GOOGL") == "USD"
    
    def test_japan_ticker(self):
        assert _guess_currency("7203.T") == "JPY"
        assert _guess_currency("6752.T") == "JPY"
    
    def test_germany_ticker(self):
        assert _guess_currency("SAP.DE") == "EUR"
        assert _guess_currency("BMW.DE") == "EUR"
        assert _guess_currency("SIE.DE") == "EUR"
    
    def test_france_ticker(self):
        assert _guess_currency("AIR.PA") == "EUR"
        assert _guess_currency("SAN.PA") == "EUR"
    
    def test_netherlands_ticker(self):
        assert _guess_currency("ADYEN.AS") == "EUR"
    
    def test_denmark_ticker(self):
        assert _guess_currency("VWS.CO") == "DKK"
        assert _guess_currency("NOVO.CO") == "DKK"
    
    def test_uk_ticker(self):
        assert _guess_currency("RR.L") == "USD"  # Default

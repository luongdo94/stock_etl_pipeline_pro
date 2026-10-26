"""Provider-to-TradingView symbol mapping across exchanges."""
import pytest

from core.symbols import get_tv_symbol


@pytest.mark.parametrize("yahoo,tv", [("BRK-B", "BRK.B"), ("BF-B", "BF.B"), ("BT-A.L", "LSE:BT.A"),
                                      ("0285.HK", "HKEX:285"), ("0005.HK", "HKEX:5"), ("AAPL", "AAPL"),
                                      ("SIE.DE", "XETR:SIE"), ("DX-Y.NYB", "DX-Y.NYB"),
                                      ("NDA-FI.HE", "OMXHEX:NDA_FI"), ("2330.TW", "TWSE:2330"), ("VOLV-B.ST", "OMXSTO:VOLV_B"),
                                      ("PETR4.SA", "BMFBOVESPA:PETR4"), ("RELIANCE.NS", "NSE:RELIANCE"),
                                      ("UCB.BR", "EURONEXT:UCB"), ("3529.TWO", "TPEX:3529"),
                                      ("RR.L", "LSE:RR."), ("BP.L", "LSE:BP."), ("GSK.L", "LSE:GSK"), ("BAJAJ-AUTO.BO", "BSE:BAJAJ_AUTO")])
def test_tradingview_symbols(yahoo, tv):
    # K7-06, R7-06: exchange-specific symbols share one table.
    assert get_tv_symbol(yahoo) == tv

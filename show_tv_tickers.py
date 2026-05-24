from etl.extract import TICKERS
print("\n" + "="*50)
print("🎯 DANH SÁCH MÃ ĐƯỢC PHÁT HIỆN TỪ TRADINGVIEW")
print("="*50)
count = 0
for ticker, meta in TICKERS.items():
    if 'discovery_source' in meta:
        count += 1
        print(f"[{count}] {ticker:10s} | {meta['discovery_source']:20s} | {meta['name']}")
print(f"\nTổng cộng: {count} siêu cổ phiếu mới.\n")

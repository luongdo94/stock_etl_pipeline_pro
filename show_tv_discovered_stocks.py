"""
Script để hiển thị các mã cổ phiếu được phát hiện qua TradingView filters
"""
import sys
import os

# Add etl to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etl.extract import fetch_dynamic_tv_tickers, load_tickers_config
import pandas as pd

print("\n" + "="*80)
print("🔮 ĐANG FETCH DỮ LIỆU TỪ TRADINGVIEW...")
print("="*80)

# Load base tickers
base_tickers = load_tickers_config()
print(f"📊 Base tickers trong config: {len(base_tickers)} mã")

# Fetch TradingView tickers
try:
    tv_tickers = fetch_dynamic_tv_tickers(base_tickers)
    
    if not tv_tickers:
        print("\n" + "="*80)
        print("📭 KHÔNG CÓ MÃ NÀO TỪ TRADINGVIEW")
        print("="*80)
        print("\nLý do có thể:")
        print("  1. TradingView API không trả về kết quả (rate limit)")
        print("  2. Tất cả các mã đã tồn tại trong base_tickers")
        print("  3. Các mã bị lọc ra do duplicate hoặc không đủ điều kiện")
        exit(0)
    
    print(f"✅ TradingView phát hiện: {len(tv_tickers)} mã mới\n")
    
    # Convert to DataFrame for better display
    data = []
    for ticker, meta in tv_tickers.items():
        data.append({
            'Ticker': ticker,
            'Company': meta.get('name', 'N/A')[:45],
            'Sector': meta.get('sector', 'N/A')[:20],
            'Region': meta.get('region', 'N/A'),
            'Filter': meta.get('discovery_source', 'N/A').replace('TV_', '')
        })
    
    df = pd.DataFrame(data)
    
    # Group by filter
    print("="*80)
    print("📊 PHÂN LOẠI THEO BỘ LỌC TRADINGVIEW")
    print("="*80)
    
    filter_map = {
        'VALUE_STOCKS': '💎 Value Stocks (P/E<15, P/B<1.5, Div>2%)',
        'GROWTH_AT_REASONABLE_PRICE': '🌱 GARP (EPS>15%, Rev>10%, P/E<25)',
        'BREAKOUT_MOMENTUM': '⚡ Breakout Momentum (MA50>MA200, RSI 60-75)',
        'QUALITY_COMPOUNDERS': '🛡️ Quality Compounders (ROIC>15%, ROE>20%)',
        'HIGH_YIELD_DIVIDEND': '💰 High Yield Dividend (Yield>4%, Payout<60%)'
    }
    
    for filter_code, filter_name in filter_map.items():
        filter_df = df[df['Filter'] == filter_code]
        if not filter_df.empty:
            print(f"\n{filter_name}")
            print(f"Số lượng: {len(filter_df)} mã")
            print("-" * 80)
            for _, row in filter_df.iterrows():
                print(f"  {row['Ticker']:12} | {row['Company']:45} | {row['Sector']:20} | {row['Region']}")
    
    # Summary
    print("\n" + "="*80)
    print("📈 THỐNG KÊ TỔNG QUAN")
    print("="*80)
    summary = df.groupby('Filter').size().reset_index(name='Count')
    summary['Filter'] = summary['Filter'].map(lambda x: filter_map.get(x, x))
    print(summary.to_string(index=False))
    
    # Region breakdown
    print("\n" + "="*80)
    print("🌍 PHÂN BỐ THEO KHU VỰC")
    print("="*80)
    region_summary = df.groupby('Region').size().reset_index(name='Count').sort_values('Count', ascending=False)
    print(region_summary.to_string(index=False))
    
    # Sector breakdown
    print("\n" + "="*80)
    print("🏭 PHÂN BỐ THEO NGÀNH")
    print("="*80)
    sector_summary = df.groupby('Sector').size().reset_index(name='Count').sort_values('Count', ascending=False).head(10)
    print(sector_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"✅ TỔNG CỘNG: {len(tv_tickers)} mã cổ phiếu từ TradingView")
    print("="*80)
    print("\n💡 Các mã này sẽ được thêm vào database khi chạy ETL pipeline:")
    print("   python run.py --sync\n")

except Exception as e:
    print(f"\n❌ Lỗi khi fetch từ TradingView: {e}")
    import traceback
    traceback.print_exc()

#!/bin/bash

# Kích hoạt môi trường ảo nếu tồn tại
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Bắt đầu quá trình ETL Full Refresh (Tải toàn bộ dữ liệu lịch sử)..."
echo "Quá trình này có thể mất một vài phút tùy vào băng thông mạng của bạn."

# Chạy pipeline với cờ --full
python3 run.py --full

echo "✅ Đã hoàn thành tải toàn bộ dữ liệu."

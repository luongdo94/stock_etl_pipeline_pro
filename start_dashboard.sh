#!/bin/bash

# Kích hoạt môi trường ảo nếu tồn tại
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Bắt đầu khởi động Stock Dashboard trên cổng 8503..."

# Chạy app.py bằng Streamlit
streamlit run app.py --server.port 8503

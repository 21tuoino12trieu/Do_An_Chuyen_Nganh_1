# Sử dụng image Python 3.10 tối giản
FROM python:3.10-slim

# Thiết lập biến môi trường cơ bản
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết
# - build-essential, git: phục vụ build/cài đặt một số thư viện Python
# - curl: dùng cho HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng cache layer của Docker
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở port 8501 (mặc định của Streamlit)
EXPOSE 8501

# Healthcheck cho ứng dụng Streamlit
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lệnh khởi chạy ứng dụng Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


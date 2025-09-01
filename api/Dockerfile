FROM python:3.12-slim

WORKDIR /app

# системные зависимости (при необходимости openpyxl/numba/etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# скопируй зависимости; если есть requirements.txt — лучше разнести кэш-слои
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# код
COPY fastapi_api.py /app/fastapi_api.py

# переменные окружения можно пробросить из k8s/compose
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "fastapi_api:app", "--host", "0.0.0.0", "--port", "8000"]

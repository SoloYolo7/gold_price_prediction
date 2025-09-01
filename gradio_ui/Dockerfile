FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY gradio_app_finance.py /app/gradio_app_finance.py

ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "gradio_app_finance.py"]

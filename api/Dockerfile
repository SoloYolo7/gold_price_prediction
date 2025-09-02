FROM python:3.10.6-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_api.py /app/fastapi_api.py

ENV PYTHONUNBUFFERED=1

ENV ROOT_PATH=/api-gold-price-prediction

EXPOSE 8000

CMD ["sh","-lc","uvicorn fastapi_api:app --host 0.0.0.0 --port 8000 --root-path ${ROOT_PATH}"]

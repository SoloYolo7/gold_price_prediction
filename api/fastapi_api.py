import fastapi
import mlflow
import pandas as pd
import numpy as np
import io
import os
import traceback
from fastapi import UploadFile, File, HTTPException
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# --- 1. НАСТРОЙКИ ---
load_dotenv()
mlflow.set_tracking_uri("http://84.201.144.227:8000")
XGBOOST_RUN_ID = "82d0a09af0d144f3bdc3f7111ea5b099"
MODEL_URI = f"runs:/{XGBOOST_RUN_ID}/model_pipeline"
ROOT_PATH = os.getenv("ROOT_PATH", "/api-gold-price-prediction")

# Глобальный словарь для хранения модели
ml_models = {}

# --- 2. ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ МОДЕЛИ ПРИ СТАРТЕ ---
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Код, который выполнится при старте сервера
    print("--- Запуск API: начинаем загрузку модели ---")
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        ml_models["gold_predictor"] = model
        
        # Проверяем наличие схемы
        schema = model.metadata.get_input_schema()
        if schema:
            ml_models["expected_features"] = schema.input_names()
        else:
            # Если схемы нет, используем жестко заданный список
            ml_models["expected_features"] = [
                'year', 'month', 'dayofweek', 'silver close', 'oil close', 'dxy close',
                'silver close_lag1', 'silver close_roll_mean3', 'oil close_lag1',
                'oil close_roll_mean3', 'dxy close_lag1', 'dxy close_roll_mean3',
                'gold_close_lag1'
            ]
        print(f"--- Модель успешно загружена. Ожидается признаков: {len(ml_models['expected_features'])} ---")
    except Exception as e:
        print(f"--- КРИТИЧЕСКАЯ ОШИБКА: не удалось загрузить модель при старте. API будет неработоспособен. ---")
        print(traceback.format_exc())
    
    yield
    
    # Код, который выполнится при остановке сервера (очистка)
    ml_models.clear()
    print("--- Остановка API: модель выгружена. ---")


# --- 3. ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ С LIFESPAN ---
app = fastapi.FastAPI(
    title="API для предсказания цены золота",
    root_path=ROOT_PATH,          # <— из env
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,            # <— ЭТО ВАЖНО: модель загрузится на старте
)
@app.get("/")
def root():
    return {"message": "API для предсказания цены золота. Используйте эндпоинт /predict."}

@app.get("/healthz")
def healthz():
    ok = "gold_predictor" in ml_models
    return {"ok": ok}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "gold_predictor" not in ml_models:
        raise HTTPException(status_code=503, detail="Модель не была загружена при старте сервера. Сервис неработоспособен.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Неверный формат файла. Требуется .csv")

    try:
        # Предобработка данных (остается без изменений)
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        input_df = pd.read_csv(buffer)
        
        required_original_cols = ['date', 'silver close', 'oil close', 'dxy close', 'gold close']
        if not all(col in input_df.columns for col in required_original_cols):
            raise ValueError(f"Отсутствуют необходимые колонки. Требуются: {required_original_cols}")

        input_df['date'] = pd.to_datetime(input_df['date'])
        input_df = input_df.set_index('date')
        
        input_df = input_df.ffill().bfill()
        
        input_df["year"] = input_df.index.year
        input_df["month"] = input_df.index.month
        input_df["dayofweek"] = input_df.index.dayofweek
        
        key_features = ["silver close", "oil close", "dxy close"]
        for col in key_features:
            input_df[f"{col}_lag1"] = input_df[col].shift(1)
            input_df[f"{col}_roll_mean3"] = input_df[col].rolling(window=3).mean()
        
        input_df["gold_close_lag1"] = input_df["gold close"].shift(1)
        
        input_df = input_df.bfill().ffill()
        
        final_df = input_df[ml_models["expected_features"]]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка на этапе предобработки данных: {e}")

    try:
        predictions_raw = ml_models["gold_predictor"].predict(final_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка во время предсказания моделью: {e}")

    return {"predictions": predictions_raw.tolist()}
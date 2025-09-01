import fastapi
import mlflow
import pandas as pd
import numpy as np
import io
import traceback
from fastapi import UploadFile, File, HTTPException
from dotenv import load_dotenv

# --- 1. НАСТРОЙКИ ---
load_dotenv()
mlflow.set_tracking_uri("http://84.201.144.227:8000")

# ЗАМЕНИТЕ RUN_ID НА ТОТ, КОТОРЫЙ ВЫ ХОТИТЕ ИСПОЛЬЗОВАТЬ
# Например, из вашего последнего успешного запуска
XGBOOST_RUN_ID = "94ae8c757e82422c82f11493c3644ab3" # Пример, вставьте свой
MODEL_URI = f"runs:/{XGBOOST_RUN_ID}/model_pipeline" # Используем model_pipeline

# --- 2. НАДЕЖНАЯ ЗАГРУЗКА МОДЕЛИ И ПРИЗНАКОВ ---

# Создаем "железный" список признаков, на которых обучалась модель.
# Это наш запасной вариант, если схема в MLflow отсутствует.
# ВАЖНО: этот список должен ТОЧНО соответствовать тому, на чем обучалась модель.
FALLBACK_FEATURES = [
    'year', 'month', 'dayofweek', 'silver close', 'oil close', 'dxy close',
    'silver close_lag1', 'silver close_roll_mean3', 'oil close_lag1',
    'oil close_roll_mean3', 'dxy close_lag1', 'dxy close_roll_mean3',
    'gold close_lag1'
]

expected_features = []
ml_model = None

try:
    print(f"Загрузка модели из: {MODEL_URI}")
    ml_model = mlflow.pyfunc.load_model(MODEL_URI)
    
    # Пытаемся получить схему из метаданных
    schema = ml_model.metadata.get_input_schema()
    if schema:
        expected_features = schema.input_names()
        print(f"Схема модели успешно загружена. Ожидается {len(expected_features)} признаков.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Схема в модели отсутствует. Используем запасной список признаков.")
        expected_features = FALLBACK_FEATURES

except Exception as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель. {e}")
    print("API будет работать в аварийном режиме (возвращать ошибку).")
    # ml_model остается None

app = fastapi.FastAPI(title="API для предсказания цены золота")

@app.get("/")
def root():
    return {"message": "API для предсказания цены золота. Используйте эндпоинт /predict."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена на сервере. Сервис временно недоступен.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Неверный формат файла. Требуется .csv")

    try:
        # --- 3. НАДЕЖНАЯ ПРЕДОБРАБОТКА ---
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        input_df = pd.read_csv(buffer)
        
        # Проверяем наличие ключевых колонок
        required_original_cols = ['date', 'silver close', 'oil close', 'dxy close', 'gold close']
        if not all(col in input_df.columns for col in required_original_cols):
            raise ValueError(f"Отсутствуют необходимые колонки. Требуются: {required_original_cols}")

        input_df['date'] = pd.to_datetime(input_df['date'])
        input_df = input_df.set_index('date')
        
        # Заполняем пропуски
        input_df = input_df.ffill().bfill()
        
        # Создаем календарные признаки
        input_df["year"] = input_df.index.year
        input_df["month"] = input_df.index.month
        input_df["dayofweek"] = input_df.index.dayofweek
        
        # Создаем лаги и скользящие средние (как в обучении)
        key_features = ["silver close", "oil close", "dxy close"]
        for col in key_features:
            input_df[f"{col}_lag1"] = input_df[col].shift(1)
            input_df[f"{col}_roll_mean3"] = input_df[col].rolling(window=3).mean()
        
        input_df["gold_close_lag1"] = input_df["gold close"].shift(1)

        # Заполняем пропуски, появившиеся после создания признаков
        input_df = input_df.bfill().ffill()

        # Приводим датафрейм к виду, который ожидает модель
        final_df = input_df[expected_features]

    except Exception as e:
        print("--- ОШИБКА ПРЕДОБРАБОТКИ ---")
        print(traceback.format_exc())
        print("---------------------------")
        raise HTTPException(status_code=400, detail=f"Ошибка на этапе предобработки данных: {e}")

    try:
        predictions_raw = ml_model.predict(final_df)
    except Exception as e:
        print("--- ОШИБКА ПРЕДСКАЗАНИЯ ---")
        print(traceback.format_exc())
        print("-------------------------")
        raise HTTPException(status_code=422, detail=f"Ошибка во время предсказания моделью: {e}")

    return {"predictions": predictions_raw.tolist()}
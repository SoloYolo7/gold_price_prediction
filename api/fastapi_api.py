import fastapi
import mlflow
import pandas as pd
import numpy as np
import io
from fastapi import UploadFile, File, HTTPException
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri("http://84.201.144.227:8000")
XGBOOST_RUN_ID = "439c6ad7b4004723beb29f78bed50465"
MODEL_URI = f"runs:/{XGBOOST_RUN_ID}/model"

try:
    ml_model = mlflow.pyfunc.load_model(MODEL_URI)
    expected_features = ml_model.metadata.get_input_schema().input_names()
    print("Модель XGBoost и ее метаданные успешно загружены.")
except Exception as e:
    print(f"Критическая ошибка: не удалось загрузить модель. {e}")
    ml_model = None
    expected_features = []

app = fastapi.FastAPI(title="API для предсказания цены золота")


@app.get("/")
def root():
    """Корневой эндпоинт с приветственным сообщением."""
    return {"message": "API для предсказания цены золота. Используйте эндпоинт /predict."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает CSV-файл, выполняет предобработку и возвращает предсказания.
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Сервис временно недоступен.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Неверный формат файла. Требуется .csv")

    try:
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        input_df = pd.read_csv(buffer)
        
        input_df['date'] = pd.to_datetime(input_df['date'])
        input_df = input_df.set_index('date')
        
        numeric_cols = input_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            input_df[col] = input_df[col].ffill()
            input_df[col] = input_df[col].bfill()
        
        new_features_list = []
        original_cols = input_df.columns.tolist()
        
        for col in original_cols:
            new_features_list.append(input_df[col].shift(1).rename(f'{col}_lag1'))
            new_features_list.append(input_df[col].rolling(window=3).mean().rename(f'{col}_roll_mean3'))
            new_features_list.append(input_df[col].rolling(window=7).mean().rename(f'{col}_roll_mean7'))
        
        features_df = pd.concat(new_features_list, axis=1)
        
        df_full_features = pd.concat([input_df, features_df], axis=1)
        
        df_full_features.bfill(inplace=True)
        df_full_features.ffill(inplace=True)

        final_df = df_full_features.reindex(columns=expected_features, fill_value=0)

    except Exception as e:
        raise HTTPException(status_code=400, detail="Ошибка на этапе предобработки данных.")

    try:
        predictions_raw = ml_model.predict(final_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail="Ошибка во время предсказания моделью.")

    return {"predictions": predictions_raw.tolist()}
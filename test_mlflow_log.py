import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from lightgbm import LGBMRegressor
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os
from dotenv import load_dotenv

# --- ШАГ 1: ЯВНАЯ ЗАГРУЗКА И ПРОВЕРКА ПЕРЕМЕННЫХ ---
print("--- ЗАПУСК УЛУЧШЕННОГО ДИАГНОСТИЧЕСКОГО ТЕСТА ---")
print("Пытаемся загрузить переменные из файла .env...")

# Явно вызываем функцию загрузки
load_dotenv()

# Проверяем, что переменные теперь доступны внутри скрипта
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")

print("\n--- РЕЗУЛЬТАТЫ ПРОВЕРКИ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ---")
print(f"AWS_ACCESS_KEY_ID: {aws_key}")
# Для безопасности выведем только часть секретного ключа
print(f"AWS_SECRET_ACCESS_KEY: {'*' * (len(aws_secret) - 4) + aws_secret[-4:] if aws_secret else None}")
print(f"MLFLOW_S3_ENDPOINT_URL: {s3_endpoint}")
print("-------------------------------------------------")

if not all([aws_key, aws_secret, s3_endpoint]):
    print("\n!!! КРИТИЧЕСКАЯ ОШИБКА: Одна или несколько переменных не были загружены из .env файла.")
    print("Проверьте, что файл .env находится в той же папке, что и скрипт, и не содержит ошибок.")
    exit() # Прерываем выполнение, если ключей нет

# --- ШАГ 2: ПОПЫТКА ЛОГИРОВАНИЯ (только если ключи нашлись) ---

# --- НАСТРОЙКИ MLFLOW ---
mlflow.set_tracking_uri("http://84.201.144.227:8000")
mlflow.set_experiment("financial_timeseries_regression")

# --- Создаем простейшие данные ---
X_train = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
y_train = pd.Series([10, 20])

# --- Обучаем простую модель ---
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
print("\nМодель обучена.")

# --- Пытаемся залогировать модель ---
with mlflow.start_run(run_name="Diagnostic_Test_Verbose"):
    print("Попытка логирования простой модели...")
    try:
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_pipeline",
            signature=signature,
            input_example=X_train.iloc[:1]
        )
        print("\n--- УСПЕХ! Простая модель залогирована корректно. ---")

    except Exception as e:
        print(f"\n--- ОШИБКА! Не удалось залогировать простую модель. Причина: {e} ---")
        print("Это означает, что ключи были найдены, но они неверные или S3 endpoint недоступен.")
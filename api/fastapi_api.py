import io
import os
import numpy as np
import pandas as pd
import fastapi
from fastapi import UploadFile, File, HTTPException

from dotenv import load_dotenv
import mlflow

load_dotenv()

# === MLflow / модель ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://84.201.144.227:8000")
RUN_ID = os.getenv("RUN_ID", "94ae8c757e82422c82f11493c3644ab3")  # твой рабочий run
MODEL_URI = f"runs:/{RUN_ID}/model_pipeline"  # именно model_pipeline, а не "model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    ml_model = mlflow.pyfunc.load_model(MODEL_URI)
    input_schema = ml_model.metadata.get_input_schema()
    ORDERED_COLS = [c.name for c in input_schema.inputs]
    REQUIRED_COLS = [c.name for c in input_schema.inputs if c.required]
    OPTIONAL_COLS = [c.name for c in input_schema.inputs if not c.required]
    print("Модель и сигнатура успешно загружены.")
except Exception as e:
    print(f"Критическая ошибка: не удалось загрузить модель: {e}")
    ml_model = None
    ORDERED_COLS, REQUIRED_COLS, OPTIONAL_COLS = [], [], []

app = fastapi.FastAPI(title="API для предсказания цены золота")


@app.get("/")
def root():
    return {
        "message": "API для предсказания цены золота. Используйте /predict.",
        "model_uri": MODEL_URI,
        "required_features": REQUIRED_COLS,
        "optional_features": OPTIONAL_COLS,
        "total_features": len(ORDERED_COLS),
    }


def _coerce_types_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим типы столбцов и порядок строго под сигнатуру модели."""
    # integer/double из сигнатуры
    dtype_map = {c.name: str(c.type) for c in input_schema.inputs}

    # привести типы
    for col, t in dtype_map.items():
        if col not in df.columns:
            continue
        if t == "integer":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif t in ("double", "float"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # прочее оставляем как есть

    # переупорядочить колонки
    df = df.reindex(columns=ORDERED_COLS)

    # возможные NaN после лагов на первых строках
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


def _ensure_required(df: pd.DataFrame):
    """Проверяем, что все обязательные фичи присутствуют."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Отсутствуют обязательные колонки: {missing}")


def _maybe_add_calendar(df: pd.DataFrame, original: pd.DataFrame):
    """Если в CSV есть столбец date — добавляем year/month/dayofweek (если их нет)."""
    if "date" in original.columns:
        try:
            d = pd.to_datetime(original["date"])
            if "year" not in df.columns:      df["year"] = d.dt.year
            if "month" not in df.columns:     df["month"] = d.dt.month
            if "dayofweek" not in df.columns: df["dayofweek"] = d.dt.dayofweek
        except Exception:
            # не фейлим — просто пропускаем (может быть уже даны числовые колонки)
            pass


def _maybe_add_needed_lags(df: pd.DataFrame):
    """
    Добавляем ТОЛЬКО те лаги/скользящие, которые есть в сигнатуре:
      - silver close_lag1, silver close_roll_mean3
      - oil close_lag1,    oil close_roll_mean3
      - gold_close_lag1
    Если исходные базовые колонки есть.
    """
    # helper: безопасный shift/roll
    def add_shift(src_col: str, new_col: str):
        if new_col in ORDERED_COLS and new_col not in df.columns and src_col in df.columns:
            df[new_col] = df[src_col].shift(1)

    def add_roll3(src_col: str, new_col: str):
        if new_col in ORDERED_COLS and new_col not in df.columns and src_col in df.columns:
            df[new_col] = df[src_col].rolling(window=3).mean()

    add_shift("silver close", "silver close_lag1")
    add_roll3("silver close", "silver close_roll_mean3")

    add_shift("oil close", "oil close_lag1")
    add_roll3("oil close", "oil close_roll_mean3")

    add_shift("gold close", "gold_close_lag1")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает CSV с исходными признаками (как на обучении),
    допускается наличие столбца date — календарные признаки посчитаем.
    Лаги/скользящие добавим только те, которые требуются сигнатурой.
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Сервис временно недоступен.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Неверный формат файла. Требуется .csv")

    # читаем CSV
    try:
        raw = await file.read()
        df_in = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать CSV.")

    # базовая подготовка: календарные + нужные лаги
    try:
        df = df_in.copy()
        _maybe_add_calendar(df, df_in)
        _maybe_add_needed_lags(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при генерации признаков: {e}")

    # проверка обязательных признаков
    _ensure_required(df)

    # добавляем отсутствующие опциональные столбцы (если их нет, модель всё равно ждёт их наличие)
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # выбрасываем столбец date (его нет в сигнатуре)
    if "date" in df.columns and "date" not in ORDERED_COLS:
        df = df.drop(columns=["date"])

    # приведение типов и порядок колонок
    try:
        df_ready = _coerce_types_and_order(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось привести вход к сигнатуре модели: {e}")

    # предсказание
    try:
        preds = ml_model.predict(df_ready)
        preds = np.asarray(preds).reshape(-1).tolist()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка во время предсказания: {e}")

    return {"predictions": preds, "rows": len(preds)}
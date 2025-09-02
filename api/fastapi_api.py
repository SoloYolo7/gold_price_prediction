import fastapi
import mlflow
import pandas as pd
import numpy as np
import io
import traceback
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# --- 1. НАСТРОЙКИ ---
load_dotenv()
mlflow.set_tracking_uri("http://84.201.144.227:8000")

# >>> ОСТАВЛЯЕМ ВАШ WIDE-RUN <<<
XGBOOST_RUN_ID = "82d0a09af0d144f3bdc3f7111ea5b099"
MODEL_URI = f"runs:/{XGBOOST_RUN_ID}/model_pipeline"

# Глобальное хранилище
ml_models = {}  # {"model":..., "features_all": [...], "features_required":[...], "features_optional":[...]}

def _parse_schema(pyfunc_model) -> dict:
    """Разобрать схему входов MLflow: required/optional."""
    schema = pyfunc_model.metadata.get_input_schema()
    features_all, features_required, features_optional = [], [], []
    if schema:
        for col in schema.inputs:
            name = col.name
            features_all.append(name)
            # у объекта ColumnSchema есть is_optional
            if getattr(col, "is_optional", False):
                features_optional.append(name)
            else:
                features_required.append(name)
    return {
        "all": features_all,
        "required": features_required,
        "optional": features_optional,
    }

def _compute_engineered(df: pd.DataFrame, target_cols: list):
    """Посчитать производные признаки, если можем."""
    # year/month/dayofweek — если есть 'date'
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if "year" in target_cols and "year" not in df.columns:
            df["year"] = dt.dt.year
        if "month" in target_cols and "month" not in df.columns:
            df["month"] = dt.dt.month
        if "dayofweek" in target_cols and "dayofweek" not in df.columns:
            df["dayofweek"] = dt.dt.dayofweek

    # Лаги и скользящие окна для silver/oil/gold — если исходники есть
    if "silver close" in df.columns:
        if "silver close_lag1" in target_cols and "silver close_lag1" not in df.columns:
            df["silver close_lag1"] = df["silver close"].shift(1)
        if "silver close_roll_mean3" in target_cols and "silver close_roll_mean3" not in df.columns:
            df["silver close_roll_mean3"] = df["silver close"].rolling(window=3).mean()

    if "oil close" in df.columns:
        if "oil close_lag1" in target_cols and "oil close_lag1" not in df.columns:
            df["oil close_lag1"] = df["oil close"].shift(1)
        if "oil close_roll_mean3" in target_cols and "oil close_roll_mean3" not in df.columns:
            df["oil close_roll_mean3"] = df["oil close"].rolling(window=3).mean()

    if "gold close" in df.columns:
        if "gold_close_lag1" in target_cols and "gold_close_lag1" not in df.columns:
            df["gold_close_lag1"] = df["gold close"].shift(1)

    # добить пропуски после лагов/роллингов
    df = df.ffill().bfill()
    return df

# --- 2. LIFE-SPAN: грузим модель и схему ---
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("--- Запуск API: загрузка модели ---")
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        parsed = _parse_schema(model)
        ml_models["model"] = model
        ml_models["features_all"] = parsed["all"]
        ml_models["features_required"] = parsed["required"]
        ml_models["features_optional"] = parsed["optional"]
        print(f"Модель загружена. Всего признаков: {len(parsed['all'])}, "
              f"обязательных: {len(parsed['required'])}, опциональных: {len(parsed['optional'])}")
    except Exception:
        print("КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ МОДЕЛИ:\n", traceback.format_exc())
    yield
    ml_models.clear()
    print("--- Остановка API ---")

# --- 3. APP ---
app = fastapi.FastAPI(
    title="API для предсказания цены золота",
    root_path="/api-gold-price-prediction",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {"message": "OK. Используйте /predict, /schema, /template."}

@app.get("/healthz")
def healthz():
    ok = "model" in ml_models
    return {"ok": ok}

# >>> Новый: вернуть схему <<<
@app.get("/schema")
def get_schema():
    if "model" not in ml_models:
        raise HTTPException(503, "Модель не загружена.")
    return {
        "features_all": ml_models["features_all"],
        "features_required": ml_models["features_required"],
        "features_optional": ml_models["features_optional"],
        "note": "Обязательные столбцы должны присутствовать в CSV. Опциональные можно не передавать — мы создадим пустые."
    }

# >>> Новый: CSV-шаблон <<<
@app.get("/template")
def get_template():
    if "model" not in ml_models:
        raise HTTPException(503, "Модель не загружена.")
    cols = ml_models["features_all"]
    # пустой шаблон на 3 строки
    df = pd.DataFrame(columns=cols, data=[[""]*len(cols) for _ in range(3)])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="template_gold_model.csv"'}
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "model" not in ml_models:
        raise HTTPException(503, "Модель не была загружена при старте сервера.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Неверный формат файла. Требуется .csv")

    # --- читаем CSV ---
    try:
        raw = await file.read()
        df_in = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"CSV не читается: {e}")

    # --- валидация/достройка признаков ---
    feats_all = ml_models["features_all"]
    feats_req = ml_models["features_required"]
    feats_opt = ml_models["features_optional"]

    # 1) посчитать производные, если можем
    try:
        df_in = _compute_engineered(df_in, feats_all)
    except Exception as e:
        raise HTTPException(400, f"Ошибка при вычислении производных признаков: {e}")

    # 2) обязательные колонки должны быть
    missing_required = [c for c in feats_req if c not in df_in.columns]
    if missing_required:
        raise HTTPException(
            400,
            {
                "error": "Отсутствуют обязательные колонки",
                "missing_required": missing_required,
                "hint": "Скачайте шаблон /template или запросите список /schema"
            },
        )

    # 3) опциональные — создаём с NaN, если их нет
    for c in feats_opt:
        if c not in df_in.columns:
            df_in[c] = np.nan

    # 4) привести порядок и отдать в модель
    final_df = df_in.reindex(columns=feats_all)

    # --- инференс ---
    try:
        preds = ml_models["model"].predict(final_df)
    except Exception as e:
        raise HTTPException(422, f"Ошибка во время предсказания: {e}")

    return {"predictions": np.asarray(preds).ravel().tolist()}
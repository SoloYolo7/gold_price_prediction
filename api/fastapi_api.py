import os
import io
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import fastapi
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html

# ========= 0. Конфиг =========
load_dotenv()

API_PREFIX = "/api-gold-price-prediction"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://84.201.144.227:8000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = os.getenv("RUN_ID", "82d0a09af0d144f3bdc3f7111ea5b099")
MODEL_URI = os.getenv("MODEL_URI", f"runs:/{RUN_ID}/model_pipeline")

ml_models: Dict[str, Any] = {
    "model": None,
    "features_all": None,
    "features_required": None,
    "features_optional": None,
    "startup_error": None,
}


def _parse_schema(pyfunc_model) -> dict:
    schema = pyfunc_model.metadata.get_input_schema()
    features_all, features_required, features_optional = [], [], []
    if schema:
        for col in schema.inputs:
            name = col.name
            features_all.append(name)
            if getattr(col, "is_optional", False):
                features_optional.append(name)
            else:
                features_required.append(name)
    return {
        "all": features_all,
        "required": features_required,
        "optional": features_optional,
    }


def _compute_engineered(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    # даты
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if "year" in target_cols and "year" not in df.columns:
            df["year"] = dt.dt.year
        if "month" in target_cols and "month" not in df.columns:
            df["month"] = dt.dt.month
        if "dayofweek" in target_cols and "dayofweek" not in df.columns:
            df["dayofweek"] = dt.dt.dayofweek

    # серебро
    if "silver close" in df.columns:
        if "silver close_lag1" in target_cols and "silver close_lag1" not in df.columns:
            df["silver close_lag1"] = df["silver close"].shift(1)
        if "silver close_roll_mean3" in target_cols and "silver close_roll_mean3" not in df.columns:
            df["silver close_roll_mean3"] = df["silver close"].rolling(window=3).mean()

    # нефть
    if "oil close" in df.columns:
        if "oil close_lag1" in target_cols and "oil close_lag1" not in df.columns:
            df["oil close_lag1"] = df["oil close"].shift(1)
        if "oil close_roll_mean3" in target_cols and "oil close_roll_mean3" not in df.columns:
            df["oil close_roll_mean3"] = df["oil close"].rolling(window=3).mean()

    # золото
    if "gold close" in df.columns:
        if "gold_close_lag1" in target_cols and "gold_close_lag1" not in df.columns:
            df["gold_close_lag1"] = df["gold close"].shift(1)

    df = df.ffill().bfill()
    return df


# ========= 1. Lifespan =========
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("--- Startup: loading model ---")
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        parsed = _parse_schema(model)
        ml_models["model"] = model
        ml_models["features_all"] = parsed["all"]
        ml_models["features_required"] = parsed["required"]
        ml_models["features_optional"] = parsed["optional"]
        ml_models["startup_error"] = None
        print(f"Model loaded OK. {len(parsed['all'])} features")
    except Exception:
        err = traceback.format_exc()
        ml_models["startup_error"] = err
        print("CRITICAL: failed to load model:\n", err)

    yield
    ml_models.clear()
    print("--- Shutdown: cleared ---")


# ========= 2. App =========
app = fastapi.FastAPI(
    title="API для предсказания цены золота",
    docs_url=None,          # docs и openapi подключаем вручную
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


# ========= 3. Эндпоинты под префиксом =========
@app.get(f"{API_PREFIX}/")
def root():
    return {"message": "Gold-price prediction API. Use /schema, /template, /predict"}


@app.get(f"{API_PREFIX}/healthz")
def healthz():
    return {"ok": ml_models.get("model") is not None}


@app.get(f"{API_PREFIX}/schema")
def schema():
    if ml_models.get("model") is None:
        raise HTTPException(503, "Модель не загружена.")
    return {
        "features_all": ml_models["features_all"],
        "features_required": ml_models["features_required"],
        "features_optional": ml_models["features_optional"],
    }


@app.get(f"{API_PREFIX}/template")
def template():
    if ml_models.get("model") is None:
        raise HTTPException(503, "Модель не загружена.")
    cols = ml_models["features_all"]
    df = pd.DataFrame(columns=cols, data=[[""] * len(cols) for _ in range(3)])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="template_gold_model.csv"'},
    )


@app.post(f"{API_PREFIX}/predict")
async def predict(file: UploadFile = File(...)):
    if ml_models.get("model") is None:
        raise HTTPException(503, "Модель не загружена.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Нужен CSV файл.")

    try:
        raw = await file.read()
        df_in = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"CSV не читается: {e}")

    feats_all = ml_models["features_all"]
    feats_req = ml_models["features_required"]
    feats_opt = ml_models["features_optional"]

    df_in = _compute_engineered(df_in, feats_all)

    missing = [c for c in feats_req if c not in df_in.columns]
    if missing:
        raise HTTPException(
            400,
            {"error": "Отсутствуют обязательные колонки", "missing_required": missing},
        )

    for c in feats_opt:
        if c not in df_in.columns:
            df_in[c] = np.nan

    final_df = df_in.reindex(columns=feats_all)

    try:
        preds = ml_models["model"].predict(final_df)
    except Exception as e:
        raise HTTPException(422, f"Ошибка предсказания: {e}")

    return {"predictions": np.asarray(preds).ravel().tolist()}


# ========= 4. Swagger и OpenAPI вручную =========
@app.get(f"{API_PREFIX}/openapi.json", include_in_schema=False)
def openapi_json():
    return JSONResponse(
        get_openapi(title=app.title, version="1.0.0", routes=app.routes)
    )


@app.get(f"{API_PREFIX}/docs", include_in_schema=False)
def swagger_ui():
    return get_swagger_ui_html(
        openapi_url=f"{API_PREFIX}/openapi.json",
        title=f"{app.title} — Docs",
    )


# ========= 5. Локальный запуск =========
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, proxy_headers=True)

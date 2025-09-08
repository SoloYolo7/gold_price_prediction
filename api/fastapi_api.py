import os
import io
import traceback
from contextlib import asynccontextmanager
from typing import Dict, List

import fastapi
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── ENV ────────────────────────────────────────────────────────────────────────
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "gold-price-prediction")
API_PREFIX = f"/api-{PROJECT_NAME}"  # внешний префикс за Ingress

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://84.201.144.227:8000")
RUN_ID = os.getenv("MLFLOW_RUN_ID", "82d0a09af0d144f3bdc3f7111ea5b099")
MODEL_URI = os.getenv("MODEL_URI", f"runs:/{RUN_ID}/model_pipeline")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ВАЖНО: НЕ задаём root_path. Роуты зарегистрируем дважды (с префиксом и без).
app = fastapi.FastAPI(
    title="API для предсказания цены золота",
    docs_url=f"{API_PREFIX}/docs",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: Dict[str, object] = {}


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
    return {"all": features_all, "required": features_required, "optional": features_optional}


def _compute_engineered(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if "year" in target_cols and "year" not in df.columns:
            df["year"] = dt.dt.year
        if "month" in target_cols and "month" not in df.columns:
            df["month"] = dt.dt.month
        if "dayofweek" in target_cols and "dayofweek" not in df.columns:
            df["dayofweek"] = dt.dt.dayofweek

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

    return df.ffill().bfill()


@asynccontextmanager
async def lifespan(_: fastapi.FastAPI):
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        parsed = _parse_schema(model)
        _state["model"] = model
        _state["features_all"] = parsed["all"]
        _state["features_required"] = parsed["required"]
        _state["features_optional"] = parsed["optional"]
        print(f"MLflow модель загружена. Признаков: {len(parsed['all'])}")
    except Exception:
        print("КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ МОДЕЛИ:\n", traceback.format_exc())
    yield
    _state.clear()


app.router.lifespan_context = lifespan


def build_router(prefix: str = "") -> APIRouter:
    r = APIRouter(prefix=prefix)

    @r.get("/")
    def root():
        return {"message": "OK. Используйте /predict, /schema, /template."}

    @r.get("/healthz")
    def healthz():
        return {"ok": "model" in _state}

    @r.get("/schema")
    def schema():
        if "model" not in _state:
            raise HTTPException(503, "Модель не загружена.")
        return {
            "features_all": _state["features_all"],
            "features_required": _state["features_required"],
            "features_optional": _state["features_optional"],
        }

    @r.get("/template")
    def template():
        if "model" not in _state:
            raise HTTPException(503, "Модель не загружена.")
        cols = list(_state["features_all"])
        df = pd.DataFrame(columns=cols, data=[[""] * len(cols) for _ in range(3)])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="template.csv"'},
        )

    @r.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if "model" not in _state:
            raise HTTPException(503, "Модель не загружена при старте сервера.")
        if not file.filename.endswith(".csv"):
            raise HTTPException(400, "Неверный формат файла. Требуется .csv")
        try:
            raw = await file.read()
            df_in = pd.read_csv(io.StringIO(raw.decode("utf-8")))
        except Exception as e:
            raise HTTPException(400, f"CSV не читается: {e}")

        feats_all = _state["features_all"]
        feats_req = _state["features_required"]
        feats_opt = _state["features_optional"]

        # производные
        try:
            df_in = _compute_engineered(df_in, feats_all)
        except Exception as e:
            raise HTTPException(400, f"Ошибка при вычислении производных признаков: {e}")

        # обязательные
        miss = [c for c in feats_req if c not in df_in.columns]
        if miss:
            raise HTTPException(400, {"error": "Отсутствуют обязательные колонки", "missing_required": miss})

        # опциональные
        for c in feats_opt:
            if c not in df_in.columns:
                df_in[c] = np.nan

        final_df = df_in.reindex(columns=feats_all)

        try:
            preds = _state["model"].predict(final_df)
        except Exception as e:
            raise HTTPException(422, f"Ошибка во время предсказания: {e}")

        return {"predictions": np.asarray(preds).ravel().tolist()}

    return r


# Без префикса — для внутри-кластерных клиентов (UI)
app.include_router(build_router(""))

# С префиксом — для внешнего доступа через Ingress
app.include_router(build_router(API_PREFIX))

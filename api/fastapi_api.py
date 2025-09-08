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
from fastapi import UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── ENV / базовые настройки ───────────────────────────────────────────────────
load_dotenv()  # .env рядом с файлом
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://84.201.144.227:8000")
RUN_ID = os.getenv("MLFLOW_RUN_ID", "82d0a09af0d144f3bdc3f7111ea5b099")
MODEL_URI = os.getenv("MODEL_URI", f"runs:/{RUN_ID}/model_pipeline")

PROJECT_NAME = os.getenv("PROJECT_NAME", "gold-price-prediction")
# внешний префикс за ингрессом:
EXT_PREFIX = os.getenv("API_PREFIX", f"/api-{PROJECT_NAME}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ВАЖНО: НЕ задаём root_path в FastAPI!
# Иначе придётся всегда слать префикс даже изнутри кластера.
app = fastapi.FastAPI(
    title="API для предсказания цены золота",
    docs_url=f"{EXT_PREFIX}/docs",          # Swagger будет доступен только по внешнему пути
    openapi_url=f"{EXT_PREFIX}/openapi.json"
)

# Разрешим CORS для UI (на всякий случай)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Глобальные артефакты модели
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
    # Грузим модель при старте
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        parsed = _parse_schema(model)
        _state.update(
            model=model,
            features_all=parsed["all"],
            features_required=parsed["required"],
            features_optional=parsed["optional"],
        )
        print(f"Модель загружена. Признаков: {len(parsed['all'])}")
    except Exception:
        print("КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ МОДЕЛИ:\n", traceback.format_exc())
    yield
    _state.clear()

app.router.lifespan_context = lifespan  # привяжем lifespan

# ── регистрируем одни и те же эндпоинты на двух префиксах ─────────────────────
from fastapi import APIRouter

def make_router(prefix: str = "") -> APIRouter:
    router = APIRouter(prefix=prefix)

    @router.get("/", tags=["internal"] if prefix == "" else None)
    def root():
        return {"message": "OK. Используйте /predict, /schema, /template."}

    @router.get("/healthz", tags=["internal"] if prefix == "" else None)
    def healthz():
        return {"ok": "model" in _state}

    @router.get("/debug/startup-error")
    def startup_error():
        if "model" not in _state:
            return JSONResponse({"error": "model_not_loaded"}, status_code=503)
        return {"ok": True}

    @router.get("/schema")
    def get_schema():
        if "model" not in _state:
            raise HTTPException(503, "Модель не загружена.")
        return {
            "features_all": _state["features_all"],
            "features_required": _state["features_required"],
            "features_optional": _state["features_optional"],
            "note": "Обязательные столбцы должны присутствовать в CSV. Опциональные создадим автоматически.",
        }

    @router.get("/template")
    def get_template():
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

    @router.post("/predict")
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

        try:
            df_in = _compute_engineered(df_in, feats_all)
        except Exception as e:
            raise HTTPException(400, f"Ошибка при вычислении производных признаков: {e}")

        missing_required = [c for c in feats_req if c not in df_in.columns]
        if missing_required:
            raise HTTPException(
                400,
                {
                    "error": "Отсутствуют обязательные колонки",
                    "missing_required": missing_required,
                    "hint": "Скачайте шаблон /template или запросите список /schema",
                },
            )

        for c in feats_opt:
            if c not in df_in.columns:
                df_in[c] = np.nan

        final_df = df_in.reindex(columns=feats_all)

        try:
            preds = _state["model"].predict(final_df)
        except Exception as e:
            raise HTTPException(422, f"Ошибка во время предсказания: {e}")

        return {"predictions": np.asarray(preds).ravel().tolist()}

    return router

# Роутер БЕЗ префикса (внутрикластерные клиенты: http://api-.../predict)
app.include_router(make_router(prefix=""))

# Роутер С префиксом (внешний доступ через Ingress: https://.../api-.../predict)
app.include_router(make_router(prefix=EXT_PREFIX))

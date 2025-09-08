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
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.openapi.utils import get_openapi

# ===================== 0) Конфиг =====================
load_dotenv()

API_PREFIX = "/api-gold-price-prediction"  # ваш внешний префикс из Ingress ALB (без rewrite)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://84.201.144.227:8000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = os.getenv("RUN_ID", "82d0a09af0d144f3bdc3f7111ea5b099")
MODEL_URI = os.getenv("MODEL_URI", f"runs:/{RUN_ID}/model_pipeline")  # по скрину — model_pipeline

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

    # добивка пропусков
    df = df.ffill().bfill()
    return df


# ===================== 1) Lifespan =====================
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
        print(f"Model loaded OK from {MODEL_URI}. features={len(parsed['all'])}")
    except Exception:
        err = traceback.format_exc()
        ml_models["startup_error"] = err
        print("CRITICAL: model load failed:\n", err)

    yield

    ml_models.clear()
    print("--- Shutdown: cleared in-memory model ---")


# ===================== 2) Приложение =====================
# ВНИМАНИЕ: docs/openapi объявим вручную на нужном префиксе, чтобы не было двойных префиксов
app = fastapi.FastAPI(
    title="API для предсказания цены золота",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# Пробные эндпоинты для kube-проб (идут напрямую в pod, без префикса)
@app.get("/", tags=["internal"])
def _root_probe():
    return {"message": "OK"}

@app.get("/livez", tags=["internal"])
def _livez():
    return {"ok": True}


# ===================== 3) Публичные эндпоинты (с префиксом) =====================
@app.get(f"{API_PREFIX}/")
def root():
    return {
        "message": "Gold-price prediction API",
        "endpoints": {
            "healthz": f"{API_PREFIX}/healthz",
            "schema": f"{API_PREFIX}/schema",
            "template": f"{API_PREFIX}/template",
            "predict": f"{API_PREFIX}/predict",
            "docs": f"{API_PREFIX}/docs",
            "openapi": f"{API_PREFIX}/openapi.json",
            "startup-error": f"{API_PREFIX}/debug/startup-error",
        },
    }

@app.get(f"{API_PREFIX}/healthz")
def healthz():
    ok = ml_models.get("model") is not None
    return {"ok": ok}

@app.get(f"{API_PREFIX}/debug/startup-error", response_class=PlainTextResponse)
def startup_error():
    err = ml_models.get("startup_error")
    return err or "No startup error."

@app.get(f"{API_PREFIX}/schema")
def get_schema():
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Модель не загружена.")
    return {
        "features_all": ml_models["features_all"],
        "features_required": ml_models["features_required"],
        "features_optional": ml_models["features_optional"],
        "note": "Обязательные столбцы должны присутствовать в CSV. Опциональные можно опустить — мы создадим их с NaN.",
    }

@app.get(f"{API_PREFIX}/template")
def get_template():
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Модель не загружена.")
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
        raise HTTPException(status_code=503, detail="Модель не была загружена при старте сервера.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Неверный формат файла. Требуется .csv")

    # читаем CSV
    try:
        raw = await file.read()
        df_in = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV не читается: {e}")

    feats_all: List[str] = ml_models["features_all"]
    feats_req: List[str] = ml_models["features_required"]
    feats_opt: List[str] = ml_models["features_optional"]

    # вычислим производные
    try:
        df_in = _compute_engineered(df_in, feats_all)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при вычислении производных признаков: {e}")

    # обязательные колонки
    missing_required = [c for c in feats_req if c not in df_in.columns]
    if missing_required:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Отсутствуют обязательные колонки",
                "missing_required": missing_required,
                "hint": f"Скачайте шаблон {API_PREFIX}/template или запросите список {API_PREFIX}/schema",
            },
        )

    # опциональные — создаём
    for c in feats_opt:
        if c not in df_in.columns:
            df_in[c] = np.nan

    # порядок колонок
    final_df = df_in.reindex(columns=feats_all)

    # инференс
    try:
        preds = ml_models["model"].predict(final_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка во время предсказания: {e}")

    return {"predictions": np.asarray(preds).ravel().tolist()}


# ===================== 4) Swagger / OpenAPI строго под префиксом =====================
@app.get(f"{API_PREFIX}/openapi.json", include_in_schema=False)
def openapi_json():
    return JSONResponse(get_openapi(title=app.title, version="1.0.0", routes=app.routes))

@app.get(f"{API_PREFIX}/docs", include_in_schema=False)
def swagger_ui():
    # Минимальная HTML-страница, подключающая Swagger-UI из CDN
    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{app.title} — Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css" />
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
    <script>
      window.onload = () => {{
        window.ui = SwaggerUIBundle({{
          url: "{API_PREFIX}/openapi.json",
          dom_id: '#swagger-ui'
        }});
      }};
    </script>
  </body>
</html>"""
    return fastapi.responses.HTMLResponse(html)


# ===================== 5) Локальный запуск =====================
# uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, proxy_headers=True, reload=False)
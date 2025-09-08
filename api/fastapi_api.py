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
from fastapi import UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse


# ========= 0) Конфиг =========
load_dotenv()

# Префикс, под которым сервис опубликован через ALB Ingress:
# например, "/api-gold-price-prediction"
API_PREFIX: str = os.getenv("API_PREFIX", "/api-gold-price-prediction").rstrip("/") or "/api-gold-price-prediction"

# MLflow
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://84.201.144.227:8000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ВАЖНО: Точный путь к артефакту (чаще всего "runs:/<RUN_ID>/model")
# Если у вас действительно папка называется "model_pipeline", оставьте её.
RUN_ID: str = os.getenv("RUN_ID", "82d0a09af0d144f3bdc3f7111ea5b099")
MODEL_SUBPATH: str = os.getenv("MODEL_SUBPATH", "model_pipeline")  # или "model"
MODEL_URI: str = os.getenv("MODEL_URI", f"runs:/{RUN_ID}/{MODEL_SUBPATH}")


# Хранилище модели и сопутствующих данных в памяти процесса
ml_models: Dict[str, Any] = {
    "model": None,
    "features_all": None,
    "features_required": None,
    "features_optional": None,
    "startup_error": None,
}


def _parse_schema(pyfunc_model) -> dict:
    """
    Разобрать схему входов MLflow pyfunc-модели на обязательные/опциональные.
    """
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
    """
    Посчитать производные признаки, если можем (year/month/dayofweek, лаги и rolling).
    """
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

    # добивка пропусков от лагов/роллингов
    df = df.ffill().bfill()
    return df


# ========= 1) Lifespan (стартап/шатаун) =========
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Загрузка модели при старте
    print("--- Startup: loading MLflow model ---")
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        parsed = _parse_schema(model)
        ml_models["model"] = model
        ml_models["features_all"] = parsed["all"]
        ml_models["features_required"] = parsed["required"]
        ml_models["features_optional"] = parsed["optional"]
        ml_models["startup_error"] = None

        print(
            f"Model loaded OK from {MODEL_URI}. "
            f"features: {len(parsed['all'])}, "
            f"required: {len(parsed['required'])}, "
            f"optional: {len(parsed['optional'])}"
        )
    except Exception:
        err = traceback.format_exc()
        ml_models["startup_error"] = err
        print("CRITICAL: model load failed:\n", err)

    yield

    # Очистка при остановке
    ml_models.clear()
    print("--- Shutdown: cleared in-memory model ---")


# ========= 2) Приложение =========
app = fastapi.FastAPI(
    title="API для предсказания цены золота",
    docs_url=f"{API_PREFIX}/docs",
    openapi_url=f"{API_PREFIX}/openapi.json",
    lifespan=lifespan,
)

# CORS по желанию (можно ограничить доменами)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хэндлеры БЕЗ префикса — для kube-проб (идут прямо в pod)
@app.get("/", tags=["internal"])
def _root_probe():
    return {"message": "OK"}

@app.get("/livez", tags=["internal"])
def _livez():
    return {"ok": True}

# ========= 3) Роутер С ПРЕФИКСОМ (важно для ALB без rewrite) =========
router = APIRouter(prefix=API_PREFIX, tags=["gold-api"])


@router.get("/")
def root():
    """
    Информационный корень под префиксом.
    """
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


@router.get("/healthz")
def healthz():
    ok = ml_models.get("model") is not None
    return {"ok": ok}


@router.get("/debug/startup-error", response_class=PlainTextResponse)
def startup_error():
    """
    Вернуть текст ошибки загрузки модели (если была) для быстрой диагностики.
    """
    err = ml_models.get("startup_error")
    return err or "No startup error."


@router.get("/schema")
def get_schema():
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Модель не загружена.")
    return {
        "features_all": ml_models["features_all"],
        "features_required": ml_models["features_required"],
        "features_optional": ml_models["features_optional"],
        "note": "Обязательные столбцы должны присутствовать в CSV. Опциональные можно опустить — мы создадим их с NaN.",
    }


@router.get("/template")
def get_template():
    """
    Отдаёт CSV-шаблон с правильными колонками (3 пустые строки).
    """
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


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает CSV, дополняет вычисляемые признаки при необходимости,
    валидирует обязательные колонки и делает предсказание.
    """
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Модель не была загружена при старте сервера.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Неверный формат файла. Требуется .csv")

    # --- читаем CSV ---
    try:
        raw = await file.read()
        df_in = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV не читается: {e}")

    feats_all: List[str] = ml_models["features_all"]
    feats_req: List[str] = ml_models["features_required"]
    feats_opt: List[str] = ml_models["features_optional"]

    # 1) вычислим производные, если сможем
    try:
        df_in = _compute_engineered(df_in, feats_all)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при вычислении производных признаков: {e}")

    # 2) обязательные колонки должны быть
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

    # 3) создадим опциональные при отсутствии
    for c in feats_opt:
        if c not in df_in.columns:
            df_in[c] = np.nan

    # 4) упорядочим колонки
    final_df = df_in.reindex(columns=feats_all)

    # --- инференс ---
    try:
        preds = ml_models["model"].predict(final_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка во время предсказания: {e}")

    return {"predictions": np.asarray(preds).ravel().tolist()}


# Подключаем роутер с префиксом
app.include_router(router)


# ========= 4) Точка входа (локальный запуск) =========
# uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, proxy_headers=True, reload=False)
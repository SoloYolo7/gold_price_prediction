import os
import json
import gradio as gr
import pandas as pd
import requests
from datetime import datetime

# адрес API берём из окружения
API_URL = os.getenv("API_URL", "http://api-gold-price-prediction/predict")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

def _pretty_http_error(resp: requests.Response) -> str:
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            if "detail" in payload:
                return json.dumps(payload["detail"], ensure_ascii=False)
            return json.dumps(payload, ensure_ascii=False)
        return str(payload)
    except Exception:
        return resp.text

def predict_gold_price(uploaded_file):
    if uploaded_file is None:
        raise gr.Error("Пожалуйста, загрузите CSV файл.")

    # отправляем файл как multipart/form-data
    try:
        with open(uploaded_file.name, "rb") as f:
            files = {"file": (os.path.basename(uploaded_file.name), f, "text/csv")}
            resp = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)

        resp.raise_for_status()
        data = resp.json()
        predictions = data.get("predictions")
    except requests.exceptions.ConnectionError:
        raise gr.Error(f"Ошибка подключения к API: {API_URL}")
    except requests.exceptions.Timeout:
        raise gr.Error(f"Превышено время ожидания ответа API (timeout={REQUEST_TIMEOUT}s).")
    except requests.exceptions.HTTPError:
        raise gr.Error(f"API error {resp.status_code}: {_pretty_http_error(resp)}")
    except ValueError:
        raise gr.Error(f"API вернул не-JSON ответ: {resp.text[:500]}")
    except Exception as e:
        raise gr.Error(f"Непредвиденная ошибка при обращении к API: {e}")

    if predictions is None:
        raise gr.Error("API не вернул поле 'predictions'.")

    # формируем таблицу результата
    try:
        df_original = pd.read_csv(uploaded_file.name)
    except Exception:
        df_original = pd.DataFrame()

    if not df_original.empty and len(predictions) == len(df_original):
        df_original["Predicted_Gold_Close"] = predictions
    else:
        df_original = pd.DataFrame(predictions, columns=["Predicted_Gold_Close"])

    # сохраняем файлы для кнопок скачивания
    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("outputs", f"predictions_{ts}.csv")
    xlsx_path = csv_path.replace(".csv", ".xlsx")

    df_original.to_csv(csv_path, index=False)
    try:
        df_original.to_excel(xlsx_path, index=False, engine="openpyxl")
    except Exception:
        xlsx_path = None

    return (
        df_original,
        gr.update(visible=True, value=csv_path),
        gr.update(visible=xlsx_path is not None, value=xlsx_path or None),
    )

with gr.Blocks(theme=gr.themes.Default(), title="Предсказание цены золота") as demo:
    gr.Markdown("# 📈 Предсказание цены на золото")
    gr.Markdown(
        "Загрузите CSV с историческими данными и получите прогноз."
        f"<br/><small>API: <code>{API_URL}</code></small>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="CSV файл",
                file_types=[".csv"],
            )
            btn_predict = gr.Button("Сделать прогноз", variant="primary")
            btn_download_csv = gr.File(label="Скачать CSV", visible=False)
            btn_download_excel = gr.File(label="Скачать Excel", visible=False)
        with gr.Column(scale=2):
            df_output = gr.DataFrame(label="Результат")

    btn_predict.click(
        predict_gold_price,
        inputs=file_input,
        outputs=[df_output, btn_download_csv, btn_download_excel],
    )

if __name__ == "__main__":
    # без очереди, чтобы не было проблем с вебсокетами за ingress
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        root_path=os.getenv("ROOT_PATH", "/ui-gold-price-prediction"),
    )
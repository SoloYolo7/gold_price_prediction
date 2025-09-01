import os
import io
import json
import gradio as gr
import pandas as pd
import requests
from datetime import datetime

# Адрес API берём из окружения. По умолчанию — имя сервиса в k8s.
API_URL = os.getenv("API_URL", "/api-gold-price-prediction/predict")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

def _pretty_http_error(resp: requests.Response) -> str:
    """
    Аккуратно достаём detail из FastAPI/любого JSON-ответа при ошибке.
    Если это не JSON — возвращаем сырой текст.
    """
    try:
        payload = resp.json()
        # Попробуем несколько распространённых ключей
        if isinstance(payload, dict):
            if "detail" in payload:
                return json.dumps(payload["detail"], ensure_ascii=False)
            return json.dumps(payload, ensure_ascii=False)
        return str(payload)
    except Exception:
        return resp.text

def predict_gold_price(uploaded_file):
    if uploaded_file is None:
        raise gr.Error("Пожалуйста, загрузите CSV файл для предсказания.")

    # отправляем файл как multipart/form-data
    try:
        with open(uploaded_file.name, "rb") as f:
            files = {"file": (os.path.basename(uploaded_file.name), f, "text/csv")}
            resp = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)
        # если код не 2xx — кинет HTTPError
        resp.raise_for_status()
        data = resp.json()
        predictions = data.get("predictions")
    except requests.exceptions.ConnectionError:
        raise gr.Error(f"Ошибка подключения к API: {API_URL}")
    except requests.exceptions.Timeout:
        raise gr.Error(f"Превышено время ожидания ответа API (timeout={REQUEST_TIMEOUT}s).")
    except requests.exceptions.HTTPError:
        # Покажем тело ошибки, которое отправил FastAPI
        detail = _pretty_http_error(resp)
        raise gr.Error(f"API error {resp.status_code}: {detail}")
    except ValueError:
        # .json() не смог распарсить ответ
        raise gr.Error(f"API вернул не-JSON ответ: {resp.text[:500]}")
    except Exception as e:
        raise gr.Error(f"Непредвиденная ошибка при обращении к API: {e}")

    if predictions is None:
        raise gr.Error("API не вернул поле 'predictions'.")

    # формируем датафрейм для отображения/выгрузки
    try:
        df_original = pd.read_csv(uploaded_file.name)
    except Exception:
        # даже если вход не читается обратно (крайний случай),
        # отдадим хотя бы предсказания.
        df_original = pd.DataFrame()

    if not df_original.empty and len(predictions) == len(df_original):
        df_original["Predicted_Gold_Close"] = predictions
    else:
        df_original = pd.DataFrame(predictions, columns=["Predicted_Gold_Close"])

    # сохраняем файлы для кнопок скачивания
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"predictions_{ts}.csv")
    xlsx_path = csv_path.replace(".csv", ".xlsx")

    df_original.to_csv(csv_path, index=False)
    try:
        # openpyxl должен быть в зависимостях
        df_original.to_excel(xlsx_path, index=False, engine="openpyxl")
    except Exception:
        # если вдруг нет openpyxl — не падаем, просто не даём xlsx
        xlsx_path = None

    csv_out = gr.update(visible=True, value=csv_path)
    xlsx_out = gr.update(visible=xlsx_path is not None, value=xlsx_path if xlsx_path else None)
    return df_original, csv_out, xlsx_out


with gr.Blocks(theme=gr.themes.Default(), title="Предсказание цены золота") as demo:
    gr.Markdown("# 📈 Предсказание цены на золото")
    gr.Markdown(
        "Загрузите CSV с историческими данными и получите прогноз. "
        f"<br/><small>API: <code>{API_URL}</code></small>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="CSV файл",
                file_types=[".csv"],
                info="Для теста можно использовать sample_for_prediction.csv",
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
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        root_path=os.getenv("ROOT_PATH", "/ui-gold-price-prediction"),
    )

 
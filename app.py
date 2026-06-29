import os
import base64
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import requests

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-ultra-550b-a55b:free")
OPENROUTER_URL = "https://api.openrouter.ai/v1/chat/completions"

app = Flask(__name__)

def image_file_to_data_uri(file_storage):
    content = file_storage.read()
    mime = file_storage.mimetype or "image/jpeg"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def analyze_image_with_openrouter(image_data_uri, prompt_text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "user", "content": {"image_base64": image_data_uri}}
        ],
        "max_tokens": 800
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Lütfen bir görsel dosyası yükleyin."}), 400
    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Dosya seçilmedi."}), 400

    prompt_text = request.form.get("prompt", "").strip()
    if not prompt_text:
        prompt_text = (
            "Bu görseldeki nesneleri, sahneyi, varsa metinleri ve görselle ilgili dikkat çekici noktaları "
            "madde madde kısa ve anlaşılır şekilde açıkla. Ayrıca görselde insan yüzü varsa tahmini duygu, "
            "varsa okunabilir metinleri belirt ve görselin olası kullanım bağlamını kısaca yaz."
        )

    try:
        data_uri = image_file_to_data_uri(image_file)
        openrouter_resp = analyze_image_with_openrouter(data_uri, prompt_text)

        analysis_text = None
        try:
            choices = openrouter_resp.get("choices")
            if choices and len(choices) > 0:
                analysis_text = choices[0].get("message", {}).get("content")
        except Exception:
            analysis_text = None

        if not analysis_text:
            analysis_text = str(openrouter_resp)

        return jsonify({
            "filename": image_file.filename,
            "model_used": openrouter_resp.get("model", MODEL_NAME),
            "analysis": analysis_text
        })
    except requests.HTTPError as e:
        try:
            details = e.response.json()
        except Exception:
            details = e.response.text if e.response is not None else str(e)
        return jsonify({"error": "API hatası", "details": details}), 500
    except Exception as e:
        return jsonify({"error": "Sunucu hatası", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

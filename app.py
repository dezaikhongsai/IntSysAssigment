# === Cell 5: Deploy on web via Flask (API + Form) ===
# Chạy được trong notebook (sẽ chiếm cell) hoặc copy ra app.py để chạy độc lập.

from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import os

# ---- Config ----
MODEL_PATH = "diabetes_pipeline.sav"  # pipeline đã lưu ở Cell 4
FEATURES = ['Glucose','BMI','Age','BloodPressure','DiabetesPedigreeFunction']

# ---- Load model once ----
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy {MODEL_PATH}. Hãy chạy Cell 4 để lưu pipeline trước.")

with open(MODEL_PATH, "rb") as f:
    pipe = pickle.load(f)

app = Flask(__name__)

# ---- Simple HTML form (demo UI) ----
INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Diabetes Predictor</title>
    <style>
      body { font-family: sans-serif; max-width: 640px; margin: 40px auto; }
      .card { padding: 16px; border: 1px solid #ccc; border-radius: 10px; }
      label { display:block; margin-top: 10px; }
      input[type=number] { width: 100%; padding: 8px; }
      button { margin-top: 16px; padding: 10px 16px; }
      .result { margin-top: 16px; padding: 12px; background: #f7f7f7; border-radius: 8px; }
      small { color: #666; }
    </style>
  </head>
  <body>
    <h2>🩺 Diabetes Predictor</h2>
    <div class="card">
      <form method="post" action="/predict-form">
        <label>Glucose <input type="number" name="Glucose" step="0.01" required></label>
        <label>BMI <input type="number" name="BMI" step="0.01" required></label>
        <label>Age <input type="number" name="Age" step="1" required></label>
        <label>BloodPressure <input type="number" name="BloodPressure" step="0.01" required></label>
        <label>DiabetesPedigreeFunction <input type="number" name="DiabetesPedigreeFunction" step="0.0001" required></label>
        <button type="submit">Predict</button>
      </form>
      <div class="result">
        <small>API: POST /predict (JSON body gồm {{features}})</small>
      </div>
    </div>
  </body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(INDEX_HTML, features=FEATURES)

@app.get("/health")
def health():
    return jsonify(status="ok", model="loaded", features=FEATURES)

def _validate_and_make_df(payload: dict) -> pd.DataFrame:
    # Kiểm tra đủ feature và chuyển thành DataFrame đúng thứ tự
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return None, {"error": f"Thiếu trường: {missing}. Cần {FEATURES}"}
    try:
        row = [[float(payload[f]) for f in FEATURES]]
    except Exception as e:
        return None, {"error": f"Giá trị không hợp lệ: {e}"}
    X = pd.DataFrame(row, columns=FEATURES)
    return X, None

@app.post("/predict")
def predict_api():
    """
    Request JSON mẫu:
    {
      "Glucose": 65,
      "BMI": 70,
      "Age": 50,
      "BloodPressure": 72,
      "DiabetesPedigreeFunction": 0.5
    }
    """
    data = request.get_json(silent=True) or {}
    X, err = _validate_and_make_df(data)
    if err:
        return jsonify(err), 400

    pred = pipe.predict(X)[0]
    proba = pipe.predict_proba(X)[0, 1]
    conf = float(np.round(max(proba, 1 - proba) * 100, 2))
    label = "Diabetic" if pred == 1 else "Non-diabetic"

    return jsonify(
        prediction=int(pred),
        label=label,
        probability_positive=float(np.round(proba, 6)),
        confidence_percent=conf,
        features_order=FEATURES
    )

@app.post("/predict-form")
def predict_form():
    form = request.form.to_dict()
    X, err = _validate_and_make_df(form)
    if err:
        return jsonify(err), 400
    pred = pipe.predict(X)[0]
    proba = pipe.predict_proba(X)[0, 1]
    conf = float(np.round(max(proba, 1 - proba) * 100, 2))
    label = "Diabetic" if pred == 1 else "Non-diabetic"
    html = f"""
    <div style="font-family:sans-serif;max-width:640px;margin:40px auto;">
      <a href="/">← Back</a>
      <h3>Prediction</h3>
      <pre>Input: {form}</pre>
      <p><b>Label:</b> {label}</p>
      <p><b>Confidence:</b> {conf}%</p>
      <p><b>P(positive):</b> {proba:.6f}</p>
    </div>
    """
    return html

if __name__ == "__main__":
    # Chạy trong notebook: set use_reloader=False để tránh chạy 2 lần.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

{ "Glucose": 95,  "BMI": 24.8, "Age": 30, "BloodPressure": 72, "DiabetesPedigreeFunction": 0.20 }


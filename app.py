from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

with open("Aletheia.pkl", "rb") as f:
    model = joblib.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return jsonify({"risk_probability": prediction})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)

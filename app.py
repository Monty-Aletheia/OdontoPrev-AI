from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo salvo
with open("Aletheia.pkl", "rb") as f:
    model = joblib.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return jsonify({"risk_probability": prediction})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import os

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

SOIL_MODELS_PATH = "models/soil_prediction_models_oilseed.pkl"
SOIL_ENCODERS_PATH = "models/soil_level_encoders_oilseed.pkl"
SOIL_SCALER_PATH = "models/soil_feature_scaler_oilseed.pkl"
CROP_MODEL_PATH = "models/crop_recommender_model_oilseed.pkl"
CROP_MLB_PATH = "models/crop_mlb_oilseed.pkl"

soil_model = joblib.load(SOIL_MODELS_PATH)
soil_encoders = joblib.load(SOIL_ENCODERS_PATH)
soil_scaler = joblib.load(SOIL_SCALER_PATH)
crop_model = joblib.load(CROP_MODEL_PATH)
mlb = joblib.load(CROP_MLB_PATH)


# ---------------------------------------------------------
# SOIL PREDICTION
# ---------------------------------------------------------

def predict_soil_label(soil_features):
    X = np.array(soil_features).reshape(1, -1)
    X_scaled = soil_scaler.transform(X)
    soil_pred_idx = soil_model.predict(X_scaled)[0]
    return soil_encoders.inverse_transform([soil_pred_idx])[0]


# ---------------------------------------------------------
# CROP PREDICTION
# ---------------------------------------------------------

def recommend_crops(soil_features, top_k=5):
    X = np.array(soil_features).reshape(1, -1)
    crop_proba = crop_model.predict_proba(X)[0]
    crop_prob_list = list(zip(mlb.classes_, crop_proba))
    crop_prob_list.sort(key=lambda x: x[1], reverse=True)
    return [{"crop": c, "probability": float(p)} for c, p in crop_prob_list[:top_k]]


# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Oilseed Recommendation API is Running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        required = ["N", "P", "K", "pH", "temperature", "humidity"]
        for f in required:
            if f not in data:
                return jsonify({"error": f"Missing value: {f}"}), 400

        soil_features = [
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["pH"]),
            float(data["temperature"]),
            float(data["humidity"]),
        ]

        soil_type = predict_soil_label(soil_features)
        crops = recommend_crops(soil_features, top_k=5)

        return jsonify({"soil_type": soil_type, "recommended_crops": crops})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway uses PORT env variable
    app.run(host="0.0.0.0", port=port)

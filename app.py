from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained heatwave model
model = joblib.load("model/heatwave_model.pkl")


@app.route("/")
def home():
    return "Heatwave Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        max_temp = float(data["max_temp"])
        min_temp = float(data["min_temp"])
        humidity = float(data["humidity"])
        wind_speed = float(data["wind_speed"])
        previous_temp = float(data["previous_temp"])

        features = np.array([[
            max_temp,
            min_temp,
            humidity,
            wind_speed,
            previous_temp
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "heatwave_prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
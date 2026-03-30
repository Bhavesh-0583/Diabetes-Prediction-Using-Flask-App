from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        # Convert to numpy array
        final_features = np.array([features])

        # Apply scaling
        scaled_data = scaler.transform(final_features)

        # Prediction
        prediction = model.predict(scaled_data)

        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template('index.html', prediction_text=f'Result: {result}')

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    
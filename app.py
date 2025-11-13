from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)  # ✅ Corrected

# Load model and encoders
try:
    model = joblib.load("model_full.pkl")
    le_gender = joblib.load("encoder_gender.pkl")
    le_married = joblib.load("encoder_married.pkl")
    le_approved = joblib.load("encoder_approved.pkl")
    model_status = "Model loaded successfully"
except Exception as e:
    model_status = f"Error loading model: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract received features
        received_fields = list(data.keys())

        # ✅ Ask the model what features it was trained with
        expected_fields = getattr(model, "expected_features", None)

        if expected_fields is None:
            raise ValueError("Model not trained with feature metadata")

        # ✅ Detect unexpected or missing features
        extra_fields = [f for f in received_fields if f not in expected_fields]
        missing_fields = [f for f in expected_fields if f not in received_fields]

        if extra_fields or missing_fields:
            raise ValueError(f"Feature mismatch. Model trained with {expected_fields}, but received {received_fields}")

        # ✅ Proceed if fields match
        age = float(data.get('age'))
        income = float(data.get('income'))
        loan_amount = float(data.get('loan_amount'))
        loan_term = float(data.get('loan_term'))
        credit_score = float(data.get('credit_score'))
        gender = data.get('gender').lower()
        married = data.get('married').lower()

        gender_encoded = le_gender.transform([gender])[0]
        married_encoded = le_married.transform([married])[0]

        input_features = np.array([[age, income, loan_amount, loan_term, credit_score, gender_encoded, married_encoded]])

        # Predict
        prediction = model.predict(input_features)
        result = le_approved.inverse_transform(prediction)[0]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({
            "error": f"Model not trained or feature mismatch: {str(e)}"
        }), 400

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# -------------------------------------
# 1️⃣ Load Model and Encoders
# -------------------------------------
try:
    model = joblib.load(os.path.join("models", "model_full.pkl"))
    le_gender = joblib.load(os.path.join("models", "encoder_gender.pkl"))
    le_married = joblib.load(os.path.join("models", "encoder_married.pkl"))
    le_approved = joblib.load(os.path.join("models", "encoder_approved.pkl"))

    # Check if model has metadata
    if not hasattr(model, "expected_features"):
        model.expected_features = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'gender', 'married']

    model_status = "Model loaded successfully"

except Exception as e:
    model = None
    model_status = f"❌ Error loading model: {str(e)}"


# -------------------------------------
# 2️⃣ Home Route
# -------------------------------------
@app.route('/')
def index():
    return render_template("form.html")


# -------------------------------------
# 3️⃣ Prediction Route
# -------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ✅ Model-defined expected features
        expected_fields = getattr(model, "expected_features", None)
        if expected_fields is None:
            raise ValueError("Model not trained with feature metadata")

        received_fields = list(data.keys())
        extra_fields = [f for f in received_fields if f not in expected_fields]
        missing_fields = [f for f in expected_fields if f not in received_fields]

        # ✅ Let model metadata trigger error if schema mismatched
        if extra_fields or missing_fields:
            raise ValueError(
                f"Feature mismatch. Model trained with {expected_fields}, "
                f"but received {received_fields}"
            )

        # ✅ Extract input
        age = float(data.get('age'))
        income = float(data.get('income'))
        loan_amount = float(data.get('loan_amount'))
        loan_term = float(data.get('loan_term'))
        credit_score = float(data.get('credit_score'))
        gender = data.get('gender').lower()
        married = data.get('married').lower()

        # ✅ Encode categorical features
        try:
            gender_encoded = le_gender.transform([gender])[0]
            married_encoded = le_married.transform([married])[0]
        except Exception:
            raise ValueError("Model not trained or invalid category provided")

        # ✅ Prepare input array
        input_features = np.array([[age, income, loan_amount, loan_term, credit_score, gender_encoded, married_encoded]])

        # ✅ Predict
        try:
            prediction = model.predict(input_features)
        except ValueError as ve:
            raise ValueError(f"Model not trained or feature mismatch: {str(ve)}")

        result = le_approved.inverse_transform(prediction)[0]

        return jsonify({
            "prediction": result,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": f"Model not trained or feature mismatch: {str(e)}"
        }), 400


# -------------------------------------
# 4️⃣ Health Check Route
# -------------------------------------
@app.route('/health', methods=['GET'])
def health():
    if "successfully" in model_status:
        return jsonify({
            "status": "ok",
            "message": model_status
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": model_status
        }), 500


# -------------------------------------
# 5️⃣ Run the Flask App
# -------------------------------------
if __name__ == '__main__':
    # Host 0.0.0.0 -> accessible publicly (e.g., AWS EC2 / Docker)
    app.run(debug=False, host='0.0.0.0', port=5000)

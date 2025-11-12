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

@app.route('/')
def index():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input
        age = float(data.get('age'))
        income = float(data.get('income'))
        loan_amount = float(data.get('loan_amount'))
        loan_term = float(data.get('loan_term'))
        credit_score = float(data.get('credit_score'))
        gender = data.get('gender').lower()
        married = data.get('married').lower()

        # Encode categorical features
        gender_encoded = le_gender.transform([gender])[0]
        married_encoded = le_married.transform([married])[0]

        # Create input feature vector
        input_features = np.array([[age, income, loan_amount, loan_term, credit_score, gender_encoded, married_encoded]])

        # Predict
        prediction = model.predict(input_features)
        result = le_approved.inverse_transform(prediction)[0]

        return jsonify({"prediction": result, "input": data})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

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

if __name__ == '__main__':  # ✅ Corrected
    # Use host='0.0.0.0' to expose it on AWS EC2/public IP
    app.run(debug=False, host='0.0.0.0', port=5000)

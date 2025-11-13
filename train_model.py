from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import random
import os

# ----------------------------
# 1️⃣ Reproducibility
# ----------------------------
random.seed(42)
np.random.seed(42)

# ----------------------------
# 2️⃣ Base Data (manual samples)
# ----------------------------
raw_X = [
    [22, 40000, 15000, 12, 600, "male", "no"],
    [30, 50000, 20000, 24, 700, "female", "no"],
    [45, 80000, 30000, 36, 750, "male", "yes"],
    [23, 35000, 10000, 6, 620, "female", "no"],
    [27, 60000, 25000, 18, 670, "male", "yes"],
    [24, 50000, 15000, 12, 640, "female", "no"],
    [29, 90000, 30000, 30, 700, "male", "yes"],
    [28, 5000, 1000, 6, 700, "male", "no"],
    [26, 8000, 2000, 12, 690, "female", "yes"],
    [40, 7000, 3000, 24, 710, "female", "no"],
    [30, 9999, 30000, 24, 700, "male", "yes"],
]

# ----------------------------
# 3️⃣ Generate Synthetic Samples
# ----------------------------
genders = ["male", "female"]
married_options = ["yes", "no"]

while len(raw_X) < 1000:
    age = random.randint(18, 60)
    income = random.randint(500, 100000)
    loan_amount = random.randint(1000, 50000)
    loan_term = random.choice([6, 12, 18, 24, 30, 36])
    credit_score = random.randint(300, 850)
    gender = random.choice(genders)
    married = random.choice(married_options)

    raw_X.append([
        age, income, loan_amount, loan_term, credit_score, gender, married
    ])

# ----------------------------
# 4️⃣ Rule-Based Labels
# ----------------------------
y = []
for sample in raw_X:
    age, income, _, _, credit_score, _, _ = sample
    if age < 25 or credit_score < 600 or income <= 10000:
        y.append("denied")
    else:
        y.append("approved")

# ----------------------------
# 5️⃣ Label Encoding
# ----------------------------
genders = [row[5] for row in raw_X]
married_statuses = [row[6] for row in raw_X]

le_gender = LabelEncoder()
le_married = LabelEncoder()
le_approved = LabelEncoder()

gender_encoded = le_gender.fit_transform(genders)
married_encoded = le_married.fit_transform(married_statuses)
y_encoded = le_approved.fit_transform(y)

# ----------------------------
# 6️⃣ Build Feature Matrix
# ----------------------------
X = np.array([
    [
        row[0],             # age
        row[1],             # income
        row[2],             # loan_amount
        row[3],             # loan_term
        row[4],             # credit_score
        gender_encoded[i],  # encoded gender
        married_encoded[i], # encoded marital status
    ]
    for i, row in enumerate(raw_X)
], dtype=float)

# ----------------------------
# 7️⃣ Train Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X, y_encoded)

# ----------------------------
# 8️⃣ Store Expected Feature Metadata
# ----------------------------
feature_names = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'gender', 'married']
model.expected_features = feature_names

# ----------------------------
# 9️⃣ Save Model and Encoders
# ----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model_full.pkl")
joblib.dump(le_gender, "models/encoder_gender.pkl")
joblib.dump(le_married, "models/encoder_married.pkl")
joblib.dump(le_approved, "models/encoder_approved.pkl")

# ----------------------------
# 🔟 Verify and Print
# ----------------------------
print("✅ Model trained successfully with 1000 samples.")
print(f"📦 Model expects features: {model.expected_features}")
print("💾 Files saved in ./models/:")
print("   - model_full.pkl")
print("   - encoder_gender.pkl")
print("   - encoder_married.pkl")
print("   - encoder_approved.pkl")

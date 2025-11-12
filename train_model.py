from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import random

# Original + manually added examples
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

# Generate synthetic samples to reach 1000 total
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

# Generate labels based on your rule
y = []
for sample in raw_X:
    age, income, _, _, credit_score, _, _ = sample
    if age < 25 or credit_score < 600 or income <= 10000:
        y.append("denied")
    else:
        y.append("approved")

# Encode gender and married features
genders = [row[5] for row in raw_X]
married_statuses = [row[6] for row in raw_X]

le_gender = LabelEncoder()
le_married = LabelEncoder()

gender_encoded = le_gender.fit_transform(genders)
married_encoded = le_married.fit_transform(married_statuses)

# Build feature matrix
X = []
for i in range(len(raw_X)):
    row = raw_X[i]
    X.append([
        row[0],             # age
        row[1],             # income
        row[2],             # loan_amount
        row[3],             # loan_term
        row[4],             # credit_score
        gender_encoded[i],  # encoded gender
        married_encoded[i], # encoded marital status
    ])

X = np.array(X, dtype=float)

# Encode target labels
le_approved = LabelEncoder()
y_encoded = le_approved.fit_transform(y)

# Train the model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save model and encoders
joblib.dump(model, "model_full.pkl")
joblib.dump(le_gender, "encoder_gender.pkl")
joblib.dump(le_married, "encoder_married.pkl")
joblib.dump(le_approved, "encoder_approved.pkl")

print("✅ Model trained with 1000 samples and saved.")

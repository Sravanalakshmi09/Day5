# ==========================================
# Diabetes Risk Predictor
# VS Code Python Project
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv("diabetes.csv")

print("\nFirst Five Records")
print(data.head())

print("\nDataset Shape")
print(data.shape)

print("\nMissing Values")
print(data.isnull().sum())

# ==========================================
# Features and Target
# ==========================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy")
print(round(accuracy * 100, 2), "%")

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ==========================================
# Feature Importance
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(8,5))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# ==========================================
# Diabetes Risk Prediction
# ==========================================

print("\n========== Diabetes Risk Prediction ==========\n")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
bloodpressure = float(input("Blood Pressure: "))
skinthickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

patient = np.array([[
    pregnancies,
    glucose,
    bloodpressure,
    skinthickness,
    insulin,
    bmi,
    dpf,
    age
]])

prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

print("\n===================================")

if prediction == 1:
    print("Prediction : HIGH RISK OF DIABETES")
else:
    print("Prediction : LOW RISK OF DIABETES")

print("Risk Probability :", round(probability * 100, 2), "%")

print("===================================")

import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/features.csv")

print("Dataset Loaded Successfully!")

# ==========================
# Create Target Label
# ==========================
# High similarity = Suitable Candidate

df["Suitable"] = (df["Similarity_Score"] >= 0.05).astype(int)

# Feature
X = df[["Similarity_Score"]]

# Target
y = df["Suitable"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# XGBoost Model
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)

# Train
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report")

print(classification_report(y_test, predictions))

# Save Model
joblib.dump(model, "models/xgboost_model.pkl")

print("\nModel Saved Successfully!")
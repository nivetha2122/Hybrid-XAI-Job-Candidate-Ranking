import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/features.csv")

# Features
X = df[["Similarity_Score"]]

# Load trained model
model = joblib.load("models/xgboost_model.pkl")

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Summary Plot
shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.savefig("outputs/shap_summary.png")

print("SHAP plot saved successfully!")
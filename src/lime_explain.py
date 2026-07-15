import pandas as pd
import joblib

from lime.lime_tabular import LimeTabularExplainer

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/features.csv")

# Features used for training
X = df[["Similarity_Score"]]

# ==========================
# Load Trained Model
# ==========================

model = joblib.load("models/xgboost_model.pkl")

# ==========================
# Create LIME Explainer
# ==========================

explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=["Similarity_Score"],
    class_names=["Not Suitable", "Suitable"],
    mode="classification"
)

# ==========================
# Explain First Candidate
# ==========================

exp = explainer.explain_instance(
    X.iloc[0].values,
    model.predict_proba,
    num_features=1
)

# ==========================
# Save Explanation
# ==========================

exp.save_to_file("outputs/lime_explanation.html")

print("LIME explanation saved successfully!")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load cleaned dataset

df = pd.read_csv("data/cleaned_resume.csv")

print("Dataset Loaded Successfully!")
print(df.columns)


# Ensure Clean_Resume exists

if "Clean_Resume" not in df.columns:
    raise ValueError("Clean_Resume column not found. Run clean_text.py first.")

# Replace missing values
df["Clean_Resume"] = df["Clean_Resume"].fillna("").astype(str)

# Remove empty resumes
df = df[df["Clean_Resume"].str.strip() != ""]

print("\nNumber of resumes:", len(df))


# Job Description


job_description = """
Machine Learning Engineer

Required Skills:
Python
Machine Learning
Deep Learning
SQL
NLP
TensorFlow
Scikit-learn
XGBoost
Git
Communication
Problem Solving
"""


# TF-IDF


vectorizer = TfidfVectorizer(
    stop_words="english"
)

documents = [job_description] + df["Clean_Resume"].tolist()

tfidf_matrix = vectorizer.fit_transform(documents)

# ==========================
# Cosine Similarity
# ==========================

job_vector = tfidf_matrix[0]

resume_vectors = tfidf_matrix[1:]

similarity_scores = cosine_similarity(
    job_vector,
    resume_vectors
)

# ==========================
# Store Score
# ==========================

df["Similarity_Score"] = similarity_scores.flatten()

# ==========================
# Sort Candidates
# ==========================

df = df.sort_values(
    by="Similarity_Score",
    ascending=False
)

# Rank

df["Rank"] = range(1, len(df) + 1)


# Save


df.to_csv(
    "data/features.csv",
    index=False
)

print("\nTop 10 Candidates\n")

print(
    df[
        [
            "Rank",
            "Category",
            "Similarity_Score"
        ]
    ].head(10)
)

print("\nFeature engineering completed successfully!")

print("\nFile Saved As")

print("data/features.csv")

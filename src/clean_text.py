import pandas as pd
import re
import string

# Load dataset
df = pd.read_csv("data/Resume.csv")

# Function to clean resume text
def clean_resume(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning
df["Clean_Resume"] = df["Resume_str"].apply(clean_resume)

# Save cleaned dataset
df.to_csv("data/cleaned_resume.csv", index=False)

print("Dataset cleaned successfully!")
print(df[["Category", "Clean_Resume"]].head())
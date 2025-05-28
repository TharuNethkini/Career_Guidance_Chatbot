# intent_classifier.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\Siribaddana\OneDrive\Documents\Career-Chatbot\Career_Dataset.csv")

# Ensure there are no missing values
df = df.dropna(subset=["question", "role"])

# Use 'question' as input and 'role' as label for intent classification
questions = df["question"].tolist()
labels = df["role"].tolist()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(questions)

# Train intent classification model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, labels)

# Save the trained model and vectorizer
joblib.dump(clf, "intent_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully.")

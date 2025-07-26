import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

file_path = r"word_problems_dataset.csv"  
df = pd.read_csv(file_path, quotechar='"')   

label_col = "label"



df[label_col] = df[label_col].map({"yes": 1, "no": 0})


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["question"])


y = df[label_col]

print(pd.isna(y).sum())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, "random_forest_followup.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully.")

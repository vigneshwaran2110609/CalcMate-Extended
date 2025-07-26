import joblib


model = joblib.load("random_forest_followup.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

new_problem=["How did you solve this problem"]
# Transform input using the trained vectorizer
new_vectorized = vectorizer.transform(new_problem)

# Predict if it's a follow-up or not
prediction = model.predict(new_vectorized)
if prediction[0]==1:
    print("Not a follow up")
else:
    print("It is a follow up")
    

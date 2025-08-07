import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import clean_text

# Cache the training to speed up repeated runs
@st.cache_data
def train_model():
    # Example placeholder: replace with your real dataset
    data = pd.DataFrame({
        "text": [
            "I feel high energy and invincible", 
            "I can't get out of bed and nothing matters"
        ],
        "label": ["manic", "depressive"]
    })
    data["cleaned"] = data["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        data["cleaned"], data["label"], random_state=42
    )

    vectorizer = TfidfVectorizer()
    model = LogisticRegression()

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

st.title("Bipolar Disorder Predictor (No .pkl files needed)")

pipeline = train_model()  # Trains or caches

user_input = st.text_area("Paste a clinical interview transcript:")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        prediction = pipeline.predict([cleaned])[0]
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")

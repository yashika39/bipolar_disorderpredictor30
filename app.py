import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from preprocess import clean_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure data is available or download if not
try:
    _ = word_tokenize("Test sentence.")
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

st.title("Bipolar Disorder Predictor (with in-app NLP setup)")

@st.cache_data
def train_model():
    # Replace with your actual dataset
    texts = ["I feel on top of the world", "Everything is hopeless"]
    labels = ["manic", "depressive"]
    cleaned = [clean_text(t) for t in texts]
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])
    pipeline.fit(cleaned, labels)
    return pipeline

pipeline = train_model()

user_input = st.text_area("Paste a clinical interview transcript:")
if st.button("Predict"):
    if user_input:
        cleaned_input = clean_text(user_input)
        pred = pipeline.predict([cleaned_input])[0]
        st.success(f"Prediction: {pred}")
    else:
        st.warning("Please enter some text.")

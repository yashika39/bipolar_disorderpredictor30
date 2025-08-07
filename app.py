import streamlit as st
import joblib
from preprocess import clean_text

st.title("Bipolar Disorder Predictor from Clinical Interview")

user_input = st.text_area("Paste a clinical interview transcript:")

if st.button("Predict"):
    if user_input:
        vectorizer = joblib.load("../models/vectorizer.pkl")
        model = joblib.load("../models/LogisticRegression.pkl")  # Choose best model
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")

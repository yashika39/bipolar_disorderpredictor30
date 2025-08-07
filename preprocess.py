import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['clean_text'] = df['text'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']
    return X, y, tfidf

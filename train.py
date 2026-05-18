import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from deep_translator import GoogleTranslator

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Stopwords
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Translation function
def translate_to_english(text):

    translated = GoogleTranslator(
        source='auto',
        target='en'
    ).translate(text)

    return translated

# Streamlit UI
st.title("Multi-Language Sentiment Analysis App")

st.write("Supports English, Hindi, Urdu and more")

# User input
user_input = st.text_area("Enter text")

# Button
if st.button("Analyze"):

    # Translate text
    translated_text = translate_to_english(user_input)

    # Show translated text
    st.subheader("Translated Text")
    st.write(translated_text)

    # Clean translated text
    cleaned_text = clean_text(translated_text)

    # Vectorize
    vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vector)

    sentiment = prediction[0]

    # Show result
    st.subheader("Sentiment Result")

    if sentiment == "positive":
        st.success("Positive 😀")

    elif sentiment == "negative":
        st.error("Negative 😞")

    else:
        st.info("Neutral 😐")
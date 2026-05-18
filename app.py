import streamlit as st
import matplotlib.pyplot as plt
import sqlite3
import speech_recognition as sr
import os

from transformers import pipeline
from deep_translator import GoogleTranslator
from openai import OpenAI

import text2emotion as te

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Advanced Sentiment Analysis AI",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------------------------
# OPENAI API SETUP
# -------------------------------------------------

openai_api_key = os.getenv("OPENAI_API_KEY")

client = None

if openai_api_key:

    client = OpenAI(
        api_key=openai_api_key
    )

# -------------------------------------------------
# SIMPLE LOGIN SYSTEM
# -------------------------------------------------

USERS = {
    "azhar": {
        "name": "Azhar",
        "password": "1234"
    },
    "admin": {
        "name": "Admin",
        "password": "admin123"
    }
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "name" not in st.session_state:
    st.session_state.name = ""

if not st.session_state.logged_in:

    st.title("🔐 Login")

    username_input = st.text_input("Username")

    password_input = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username_input in USERS:

            if USERS[username_input]["password"] == password_input:

                st.session_state.logged_in = True

                st.session_state.username = username_input

                st.session_state.name = USERS[username_input]["name"]

                st.rerun()

            else:

                st.error("Incorrect password")

        else:

            st.error("User not found")

else:

    username = st.session_state.username

    name = st.session_state.name

    st.sidebar.success(f"Welcome {name}")

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.name = ""

        st.rerun()

    # -------------------------------------------------
    # DATABASE SETUP
    # -------------------------------------------------

    conn = sqlite3.connect(
        'sentiment.db',
        check_same_thread=False
    )

    cursor = conn.cursor()

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS sentiment_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT,

            original_text TEXT,

            translated_text TEXT,

            sentiment TEXT,

            confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    conn.commit()

    # -------------------------------------------------
    # LOAD AI MODEL
    # -------------------------------------------------

    classifier = pipeline(
        "sentiment-analysis"
    )

    # -------------------------------------------------
    # TRANSLATION FUNCTION
    # -------------------------------------------------

    def translate_to_english(text):

        translated = GoogleTranslator(
            source='auto',
            target='en'
        ).translate(text)

        return translated

    # -------------------------------------------------
    # APP TITLE
    # -------------------------------------------------

    st.title(
        "🤖 Advanced Sentiment Analysis AI"
    )

    st.write(
        "AI-powered multilingual sentiment and emotion analysis"
    )

    st.write(
        "Supports English, Hindi, Urdu and more"
    )

    # -------------------------------------------------
    # VOICE INPUT
    # -------------------------------------------------

    st.subheader(
        "🎤 Voice Input"
    )

    voice_text = ""

    if st.button("Start Voice Input"):

        recognizer = sr.Recognizer()

        try:

            with sr.Microphone() as source:

                st.info(
                    "Speak now..."
                )

                audio = recognizer.listen(
                    source
                )

            voice_text = recognizer.recognize_google(
                audio
            )

            st.success(
                "Voice recognized!"
            )

            st.write(
                voice_text
            )

        except:

            st.error(
                "Could not recognize voice"
            )

    # -------------------------------------------------
    # USER INPUT
    # -------------------------------------------------

    user_input = st.text_area(
        "Enter text",
        value=voice_text,
        height=150
    )

    # -------------------------------------------------
    # ANALYZE BUTTON
    # -------------------------------------------------

    if st.button("Analyze"):

        if user_input.strip() == "":

            st.warning(
                "Please enter some text"
            )

        else:

            # -----------------------------------------
            # TRANSLATE TEXT
            # -----------------------------------------

            translated_text = translate_to_english(
                user_input
            )

            st.subheader(
                "🌍 Translated Text"
            )

            st.write(
                translated_text
            )

            # -----------------------------------------
            # SENTIMENT ANALYSIS
            # -----------------------------------------

            result = classifier(
                translated_text
            )

            label = result[0]['label']

            score = result[0]['score']

            confidence = round(
                score * 100,
                2
            )

            # -----------------------------------------
            # SAVE TO DATABASE
            # -----------------------------------------

            cursor.execute(
                '''
                INSERT INTO sentiment_history (

                    username,
                    original_text,
                    translated_text,
                    sentiment,
                    confidence

                )

                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    username,
                    user_input,
                    translated_text,
                    label,
                    confidence
                )
            )

            conn.commit()

            # -----------------------------------------
            # SENTIMENT RESULT
            # -----------------------------------------

            st.subheader(
                "🧠 Sentiment Prediction"
            )

            if label == "POSITIVE":

                st.success(
                    "Positive 😀"
                )

                labels = [
                    'Positive',
                    'Negative'
                ]

                sizes = [
                    confidence,
                    100 - confidence
                ]

            else:

                st.error(
                    "Negative 😞"
                )

                labels = [
                    'Positive',
                    'Negative'
                ]

                sizes = [
                    100 - confidence,
                    confidence
                ]

            # -----------------------------------------
            # CONFIDENCE SCORE
            # -----------------------------------------

            st.subheader(
                "📊 Confidence Score"
            )

            st.write(
                f"{confidence}%"
            )

            # -----------------------------------------
            # PIE CHART
            # -----------------------------------------

            fig, ax = plt.subplots()

            ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%'
            )

            st.pyplot(
                fig
            )

            # -----------------------------------------
            # EMOTION DETECTION
            # -----------------------------------------

            st.subheader(
                "😊 Emotion Detection"
            )

            emotions = te.get_emotion(
                translated_text
            )

            st.write(
                emotions
            )

            # -----------------------------------------
            # EMOTION BAR CHART
            # -----------------------------------------

            emotion_names = list(
                emotions.keys()
            )

            emotion_values = list(
                emotions.values()
            )

            fig2, ax2 = plt.subplots()

            ax2.bar(
                emotion_names,
                emotion_values
            )

            ax2.set_ylabel(
                "Emotion Score"
            )

            st.pyplot(
                fig2
            )

            # -----------------------------------------
            # AI ASSISTANT
            # -----------------------------------------

            st.subheader(
                "🤖 AI Assistant"
            )

            if client:

                if label == "NEGATIVE":

                    prompt = f"""
                    The user feels upset.

                    User message:
                    {translated_text}

                    Respond politely and helpfully.
                    """

                else:

                    prompt = f"""
                    The user feels positive.

                    User message:
                    {translated_text}

                    Respond in a friendly way.
                    """

                try:

                    response = client.chat.completions.create(

                        model="gpt-4.1-mini",

                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    bot_reply = response.choices[0].message.content

                    st.write(
                        bot_reply
                    )

                except Exception as e:

                    st.error(
                        f"OpenAI Error: {e}"
                    )

            else:

                st.info(
                    "OpenAI API key not configured"
                )

    # -------------------------------------------------
    # USER HISTORY
    # -------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📜 Prediction History"
    )

    cursor.execute(
        '''
        SELECT
            original_text,
            sentiment,
            confidence,
            created_at

        FROM sentiment_history

        WHERE username = ?

        ORDER BY id DESC
        ''',
        (username,)
    )

    rows = cursor.fetchall()

    if rows:

        for row in rows:

            st.write(
                f"📝 Text: {row[0]}"
            )

            st.write(
                f"➡️ Sentiment: {row[1]}"
            )

            st.write(
                f"📊 Confidence: {row[2]}%"
            )

            st.write(
                f"⏰ Time: {row[3]}"
            )

            st.markdown("---")

    else:

        st.info(
            "No prediction history found"
        )

    # -------------------------------------------------
    # ADMIN DASHBOARD
    # -------------------------------------------------

    if username == 'admin':

        st.markdown("---")

        st.subheader(
            "📊 Admin Dashboard"
        )

        cursor.execute(
            '''
            SELECT COUNT(*)
            FROM sentiment_history
            '''
        )

        total = cursor.fetchone()[0]

        st.metric(
            "Total Predictions",
            total
        )

        cursor.execute(
            '''
            SELECT
                username,
                original_text,
                sentiment,
                confidence,
                created_at

            FROM sentiment_history

            ORDER BY id DESC
            '''
        )

        rows = cursor.fetchall()

        st.subheader(
            "🗂 All Predictions"
        )

        for row in rows:

            st.write(
                row
            )
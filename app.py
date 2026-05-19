# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import sqlite3
import nltk

from transformers import pipeline
from deep_translator import GoogleTranslator
from streamlit_mic_recorder import speech_to_text

import text2emotion as te

# -------------------------------------------------
# DOWNLOAD NLTK DATA
# -------------------------------------------------

nltk.download('punkt')
nltk.download('punkt_tab')

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Advanced Sentiment Analysis AI",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
        color: white;
    }

    .subtitle {
        font-size: 20px;
        color: #cfcfcf;
    }

    </style>
    """,
    unsafe_allow_html=True
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

# -------------------------------------------------
# LOGIN PAGE
# -------------------------------------------------

if not st.session_state.logged_in:

    st.title("🔐 Login")

    username_input = st.text_input(
        "Username"
    )

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

                st.error(
                    "Incorrect password"
                )

        else:

            st.error(
                "User not found"
            )

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------

else:

    username = st.session_state.username

    name = st.session_state.name

    st.sidebar.success(
        f"Welcome {name}"
    )

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
    # APP HEADER
    # -------------------------------------------------

    st.title(
        "🤖 Advanced Sentiment Analysis AI"
    )

    st.markdown(
        '<p class="subtitle">AI-powered multilingual sentiment and emotion analysis</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Supports English, Hindi, Urdu and more"
    )

    # -------------------------------------------------
    # VOICE INPUT
    # -------------------------------------------------

    st.subheader("🎤 Voice Input")

    voice_text = speech_to_text(

        language='en',

        use_container_width=True,

        just_once=True,

        key='voice'
    )

    if voice_text:

        st.success("Voice recognized!")

        st.write(voice_text)

    # -------------------------------------------------
    # USER INPUT
    # -------------------------------------------------

    user_input = st.text_area(
        "Enter text",
        value=voice_text if voice_text else "",
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

            # -------------------------------------------------
            # TRANSLATE TEXT
            # -------------------------------------------------

            translated_text = translate_to_english(
                user_input
            )

            st.subheader(
                "🌍 Translated Text"
            )

            st.success(
                translated_text
            )

            # -------------------------------------------------
            # SENTIMENT ANALYSIS
            # -------------------------------------------------

            result = classifier(
                translated_text
            )

            label = result[0]['label']

            score = result[0]['score']

            confidence = round(
                score * 100,
                2
            )

            # -------------------------------------------------
            # SAVE TO DATABASE
            # -------------------------------------------------

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

            # -------------------------------------------------
            # SENTIMENT RESULT
            # -------------------------------------------------

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

            # -------------------------------------------------
            # MODERN CONFIDENCE VISUALIZATION
            # -------------------------------------------------

            st.subheader(
                "📊 Confidence Score"
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence}%"
            )

            fig, ax = plt.subplots(
                figsize=(4, 4)
            )

            ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={
                    'width': 0.4
                }
            )

            ax.set_title(
                "Sentiment Distribution"
            )

            st.pyplot(
                fig,
                use_container_width=False
            )

            # -------------------------------------------------
            # EMOTION DETECTION
            # -------------------------------------------------

            st.subheader(
                "😊 Emotion Detection"
            )

            try:

                emotions = te.get_emotion(
                    translated_text
                )

                st.write(
                    emotions
                )

                emotion_names = list(
                    emotions.keys()
                )

                emotion_values = list(
                    emotions.values()
                )

                fig2, ax2 = plt.subplots(
                    figsize=(5, 3)
                )

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

            except Exception as e:

                st.warning(
                    "Emotion detection currently unavailable"
                )

                st.code(
                    str(e)
                )

            # -------------------------------------------------
            # AI ASSISTANT
            # -------------------------------------------------

            st.subheader(
                "🤖 AI Assistant"
            )

            if label == "NEGATIVE":

                st.error(
                    "I'm sorry you're feeling upset. Hope things improve soon. 💙"
                )

            elif label == "POSITIVE":

                st.success(
                    "That's great to hear! Keep smiling 😀"
                )

            else:

                st.info(
                    "Thank you for sharing your thoughts 🙂"
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

            st.info(
                f"""
                📝 Text: {row[0]}

                ➡️ Sentiment: {row[1]}

                📊 Confidence: {row[2]}%

                ⏰ Time: {row[3]}
                """
            )

    else:

        st.info(
            "No prediction history found"
        )

    # -------------------------------------------------
    # ADMIN DASHBOARD
    # -------------------------------------------------

    if username == "admin":

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

        admin_rows = cursor.fetchall()

        for row in admin_rows:

            st.write(row)
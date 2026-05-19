# =========================================================
# ADVANCED SENTIMENT ANALYSIS AI
# app.py
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import matplotlib.pyplot as plt
import nltk
import text2emotion as te

from deep_translator import GoogleTranslator
from streamlit_mic_recorder import speech_to_text

# =========================================================
# DOWNLOAD NLTK
# =========================================================

nltk.download('punkt')
nltk.download('stopwords')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Advanced Sentiment Analysis AI",
    page_icon="😊",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: 'Segoe UI';
}

.main {
    background: linear-gradient(to bottom right, #050816, #0B1026);
}

h1 {
    font-size: 52px !important;
    font-weight: 800 !important;
    color: white;
}

h2, h3 {
    color: white;
}

.stTextArea textarea {
    background-color: #1E293B;
    color: white;
    border-radius: 15px;
    border: 1px solid #334155;
    font-size: 18px;
}

.stButton button {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: white;
    border-radius: 14px;
    height: 55px;
    width: 180px;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton button:hover {
    transform: scale(1.03);
    transition: 0.3s;
}

.result-card {
    padding: 20px;
    border-radius: 18px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.history-card {
    background: #172554;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 15px;
    border-left: 5px solid #3B82F6;
}

.metric-box {
    background: linear-gradient(to right, #0F172A, #1E293B);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect(
    'sentiment.db',
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS sentiment_history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT,

    original_text TEXT,

    translated_text TEXT,

    sentiment TEXT,

    confidence REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()

# =========================================================
# USERS
# =========================================================

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

# =========================================================
# LOGIN PAGE
# =========================================================

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

                st.error("Incorrect password")

        else:

            st.error("User not found")

# =========================================================
# MAIN APP
# =========================================================

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

    # =========================================================
    # LOAD MODEL
    # =========================================================

    try:

        model = pickle.load(
            open('sentiment_model.pkl', 'rb')
        )

        vectorizer = pickle.load(
            open('tfidf_vectorizer.pkl', 'rb')
        )

    except Exception as e:

        st.error(
            "Model files not found."
        )

        st.code(
            str(e)
        )

        st.stop()

    # =========================================================
    # TITLE
    # =========================================================

    st.title(
        "😊 Advanced Sentiment Analysis AI"
    )

    st.write(
        "AI-powered multilingual sentiment and emotion analysis"
    )

    st.write(
        "Supports English, Hindi, Urdu and more"
    )

    # =========================================================
    # VOICE INPUT
    # =========================================================

    st.subheader(
        "🎤 Voice Input"
    )

    voice_text = speech_to_text(
        language='en',
        use_container_width=True,
        just_once=True,
        key='voice'
    )

    # =========================================================
    # TEXT INPUT
    # =========================================================

    default_text = ""

    if voice_text:
        default_text = voice_text

    user_input = st.text_area(
        "Enter text",
        value=default_text if default_text else "",
        height=180
    )

    # =========================================================
    # ANALYZE
    # =========================================================

    if st.button("Analyze"):

        if user_input.strip() == "":

            st.warning(
                "Please enter some text"
            )

        else:

            # =========================================================
            # TRANSLATION
            # =========================================================

            try:

                translated_text = GoogleTranslator(
                    source='auto',
                    target='en'
                ).translate(
                    user_input
                )

            except:

                translated_text = user_input

            st.subheader(
                "🌍 Translated Text"
            )

            st.success(
                translated_text
            )

            # =========================================================
            # PREDICTION
            # =========================================================

            transformed_text = vectorizer.transform(
                [translated_text]
            )

            prediction = model.predict(
                transformed_text
            )[0]

            probability = model.predict_proba(
                transformed_text
            )

            confidence = np.max(
                probability
            ) * 100

            label = prediction.upper()

            # =========================================================
            # SAVE HISTORY
            # =========================================================

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

            # =========================================================
            # PREDICTION RESULT
            # =========================================================

            st.subheader(
                "📌 Prediction"
            )

            if label == "POSITIVE":

                st.success(
                    f"{label} 😊"
                )

            elif label == "NEGATIVE":

                st.error(
                    f"{label} 😔"
                )

            else:

                st.info(
                    f"{label} 😐"
                )

            # =========================================================
            # CONFIDENCE SCORE
            # =========================================================

            st.subheader(
                "📈 Confidence Score"
            )

            fig, ax = plt.subplots(
                figsize=(6, 3)
            )

            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)

            ax.axis('off')

            confidence_ratio = confidence / 100

            colors = ['red', 'orange', 'gold', 'yellowgreen', 'green']

            start = -1

            for color in colors:

                ax.add_patch(
                    plt.Circle(
                        (0, 0),
                        1,
                        color=color,
                        fill=False,
                        linewidth=25
                    )
                )

            needle_x = confidence_ratio * 2 - 1

            ax.plot(
                [0, needle_x],
                [0, 0.7],
                linewidth=4
            )

            ax.text(
                0,
                -0.1,
                f"{confidence:.2f}%",
                fontsize=28,
                ha='center'
            )

            st.pyplot(fig)

            # =========================================================
            # EMOTION DETECTION
            # =========================================================

            st.subheader(
                "😊 Emotion Detection"
            )

            try:

                emotions = te.get_emotion(
                    translated_text
                )

                emotion_names = []
                emotion_values = []

                for k, v in emotions.items():

                    if v > 0:

                        emotion_names.append(k)
                        emotion_values.append(v)

                if len(emotion_names) == 0:

                    st.info(
                        "No strong emotions detected"
                    )

                else:

                    fig2, ax2 = plt.subplots(
                        figsize=(5, 5)
                    )

                    colors = [
                        "#FF6B6B",
                        "#FFD166",
                        "#06D6A0",
                        "#118AB2",
                        "#9B5DE5"
                    ]

                    wedges, texts = ax2.pie(
                        emotion_values,
                        labels=emotion_names,
                        colors=colors,
                        startangle=90,
                        wedgeprops=dict(width=0.35)
                    )

                    plt.title(
                        "The Feelings Wheel",
                        fontsize=28,
                        fontweight='bold'
                    )

                    st.pyplot(
                        fig2
                    )

            except Exception as e:

                st.warning(
                    "Emotion detection unavailable"
                )

                st.code(
                    str(e)
                )

            # =========================================================
            # AI ASSISTANT
            # =========================================================

            st.subheader(
                "🤖 AI Assistant"
            )

            if label == "NEGATIVE":

                st.write(
                    "I'm sorry you're feeling upset. Hope things improve soon ❤️"
                )

            elif label == "POSITIVE":

                st.write(
                    "That's great to hear! Keep smiling 😀"
                )

            else:

                st.write(
                    "You seem calm and neutral 😐"
                )

    # =========================================================
    # HISTORY
    # =========================================================

    with st.expander(
        "📜 Prediction History"
    ):

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

        if len(rows) == 0:

            st.info(
                "No history found"
            )

        else:

            for row in rows:

                st.markdown(
                    f"""
                    <div class="history-card">

                    <h4>📝 Text</h4>
                    <p>{row[0]}</p>

                    <h4>📌 Sentiment</h4>
                    <p>{row[1]}</p>

                    <h4>📈 Confidence</h4>
                    <p>{row[2]:.2f}%</p>

                    <h4>⏰ Time</h4>
                    <p>{row[3]}</p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =========================================================
    # ADMIN PANEL
    # =========================================================

    if username == "admin":

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

        for row in rows:

            st.write(row)
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
nltk.download('punkt_tab')

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

.main {
    background-color: #0E1117;
}

h1 {
    font-size: 52px !important;
    font-weight: bold;
}

.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 140px;
    font-size: 20px;
}

.history-card {
    background-color: #172B44;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
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
# SIMPLE LOGIN SYSTEM
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

                st.error(
                    "Incorrect password"
                )

        else:

            st.error(
                "User not found"
            )

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

    if voice_text:

        st.success(
            "Voice recognized!"
        )

        st.write(
            voice_text
        )

    # =========================================================
    # TEXT INPUT
    # =========================================================

    default_text = ""

    if voice_text:
        default_text = voice_text

    user_input = st.text_area(
        "Enter text",
        value=default_text,
        height=180
    )

    # =========================================================
    # ANALYZE BUTTON
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

            st.write(
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
            # SAVE TO DATABASE
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
            # SENTIMENT RESULT
            # =========================================================

            st.subheader(
                "📌 Prediction"
            )

            if label == "POSITIVE":

                st.success(
                    f"{label}"
                )

            else:

                st.error(
                    f"{label}"
                )

            # =========================================================
            # CONFIDENCE SCORE
            # =========================================================

            st.subheader(
                "📈 Confidence Score"
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.axis('off')

            sizes = [20, 20, 20, 20, 20]

            colors = [
                "#ff0000",
                "#ff6600",
                "#ffcc00",
                "#ccff00",
                "#33cc33"
            ]

            wedges, _ = ax.pie(
                sizes,
                radius=1.2,
                colors=colors,
                startangle=180,
                counterclock=False,
                wedgeprops=dict(
                    width=0.35,
                    edgecolor='white'
                )
            )

            circle = plt.Circle(
                (0, 0),
                0.75,
                color='white'
            )

            ax.add_artist(circle)

            angle = (180 * confidence / 100)

            x = 0.8 * np.cos(
                np.radians(180 - angle)
            )

            y = 0.8 * np.sin(
                np.radians(180 - angle)
            )

            ax.plot(
                [0, x],
                [0, y],
                lw=4,
                color='black'
            )

            ax.scatter(
                0,
                0,
                s=200,
                color='white',
                edgecolors='black',
                linewidths=3,
                zorder=5
            )

            ax.text(
                -1.2,
                -0.1,
                "LOW",
                fontsize=16,
                fontweight='bold'
            )

            ax.text(
                1.0,
                -0.1,
                "HIGH",
                fontsize=16,
                fontweight='bold'
            )

            ax.text(
                0,
                1.25,
                "CONFIDENCE",
                ha='center',
                fontsize=24,
                fontweight='bold'
            )

            ax.text(
                0,
                -0.35,
                f"{confidence:.2f}%",
                ha='center',
                fontsize=22,
                fontweight='bold'
            )

            ax.set_aspect('equal')

            st.pyplot(
                fig
            )

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
                        figsize=(8, 8)
                    )

                    ax2.pie(
                        emotion_values,
                        labels=emotion_names,
                        autopct='%1.1f%%',
                        startangle=90
                    )

                    centre_circle = plt.Circle(
                        (0, 0),
                        0.55,
                        fc='white'
                    )

                    fig2.gca().add_artist(
                        centre_circle
                    )

                    plt.title(
                        "Emotion Wheel",
                        fontsize=28,
                        fontweight='bold'
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

            # =========================================================
            # AI RESPONSE
            # =========================================================

            st.subheader(
                "🤖 AI Assistant"
            )

            if label == "NEGATIVE":

                st.write(
                    "I'm sorry you're feeling upset. Hope things improve soon ❤️"
                )

            else:

                st.write(
                    "That's great to hear! Keep smiling 😀"
                )

    # =========================================================
    # PREDICTION HISTORY
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
                "No prediction history found"
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
    # ADMIN DASHBOARD
    # =========================================================

    if username == 'admin':

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
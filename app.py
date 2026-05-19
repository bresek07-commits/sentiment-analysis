# =========================================================
# ADVANCED SENTIMENT ANALYSIS AI
# app.py
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import nltk
import text2emotion as te

from deep_translator import GoogleTranslator
from streamlit_mic_recorder import speech_to_text

# =========================================================
# DOWNLOAD NLTK
# =========================================================

nltk.download('punkt')

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
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(
        135deg,
        #020617,
        #071129,
        #020617
    );
    color: white;
}

h1 {
    font-size: 56px !important;
    font-weight: 800 !important;
    color: white !important;
}

h2, h3 {
    color: white !important;
}

.stTextArea textarea {
    background-color: #1E293B !important;
    color: white !important;
    border-radius: 18px !important;
    border: 2px solid #334155 !important;
    padding: 20px !important;
    font-size: 18px !important;
}

.stButton button {
    background: linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    );
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    height: 60px !important;
    width: 230px !important;
    font-size: 22px !important;
    font-weight: bold !important;
    transition: 0.3s !important;
}

.stButton button:hover {
    transform: scale(1.03);
}

.history-card {
    background: linear-gradient(
        135deg,
        #172554,
        #1E293B
    );
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

.prediction-positive {
    background: rgba(34,197,94,0.2);
    padding: 20px;
    border-radius: 18px;
    color: #4ADE80;
    font-size: 28px;
    font-weight: bold;
}

.prediction-negative {
    background: rgba(239,68,68,0.2);
    padding: 20px;
    border-radius: 18px;
    color: #F87171;
    font-size: 28px;
    font-weight: bold;
}

.prediction-neutral {
    background: rgba(59,130,246,0.2);
    padding: 20px;
    border-radius: 18px;
    color: #60A5FA;
    font-size: 28px;
    font-weight: bold;
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

# =========================================================
# SESSION
# =========================================================

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

    default_text = ""

    if voice_text:

        st.success(
            "Voice recognized successfully!"
        )

        default_text = voice_text

    # =========================================================
    # TEXT AREA
    # =========================================================

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
            # TRANSLATE
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
            # PREDICTION UI
            # =========================================================

            st.subheader(
                "📌 Prediction"
            )

            if label == "POSITIVE":

                st.markdown(
                    f"""
                    <div class="prediction-positive">
                        POSITIVE 😊
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif label == "NEGATIVE":

                st.markdown(
                    f"""
                    <div class="prediction-negative">
                        NEGATIVE 😔
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="prediction-neutral">
                        NEUTRAL 😐
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =========================================================
            # CONFIDENCE SCORE
            # =========================================================

            st.subheader(
                "📈 Confidence Score"
            )

            confidence_color = "#22C55E"

            if confidence < 40:
                confidence_color = "#EF4444"

            elif confidence < 70:
                confidence_color = "#F59E0B"

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(
                        135deg,
                        #172554,
                        #1E293B
                    );
                    padding:40px;
                    border-radius:25px;
                    margin-top:20px;
                    margin-bottom:30px;
                    text-align:center;
                    box-shadow:0 10px 30px rgba(0,0,0,0.4);
                ">

                    <div style="
                        color:white;
                        font-size:24px;
                        font-weight:700;
                        margin-bottom:20px;
                    ">
                        AI Prediction Confidence
                    </div>

                    <div style="
                        font-size:80px;
                        font-weight:900;
                        color:{confidence_color};
                        margin-bottom:25px;
                    ">
                        {confidence:.2f}%
                    </div>

                    <div style="
                        width:100%;
                        background:#0F172A;
                        height:24px;
                        border-radius:30px;
                        overflow:hidden;
                    ">

                        <div style="
                            width:{confidence}%;
                            height:100%;
                            background:linear-gradient(
                                90deg,
                                #3B82F6,
                                #8B5CF6,
                                #22C55E
                            );
                            border-radius:30px;
                        ">
                        </div>

                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-top:10px;
                        color:#CBD5E1;
                        font-size:14px;
                    ">
                        <span>Low Confidence</span>
                        <span>High Confidence</span>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
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

                    st.markdown(
                        """
                        <div style="
                            background:#F8FAFC;
                            padding:20px;
                            border-radius:20px;
                            text-align:center;
                        ">
                            <h1 style="
                                color:black;
                                font-size:45px;
                                font-weight:900;
                            ">
                                The Feelings Wheel
                            </h1>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.image(
                        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Plutchik-wheel.svg/800px-Plutchik-wheel.svg.png",
                        width=420
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
    # ADMIN DASHBOARD
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
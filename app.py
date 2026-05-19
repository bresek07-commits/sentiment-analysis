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
import matplotlib.pyplot as plt

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
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(
        135deg,
        #0F172A,
        #020617
    );
    color: white;
}

h1 {
    font-size: 52px !important;
    font-weight: 800 !important;
    color: white;
}

h2, h3 {
    color: white !important;
}

.stTextArea textarea {
    background-color: #1E293B !important;
    color: white !important;
    border-radius: 18px !important;
    border: 1px solid #334155 !important;
    font-size: 18px !important;
    padding: 20px !important;
}

.stButton button {
    background: linear-gradient(
        135deg,
        #2563EB,
        #7C3AED
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 15px !important;

    height: 55px !important;

    width: 170px !important;

    font-size: 20px !important;

    font-weight: 600 !important;

    transition: 0.3s !important;
}

.stButton button:hover {
    transform: scale(1.03);
}

.history-card {

    background: rgba(30,41,59,0.7);

    border: 1px solid rgba(255,255,255,0.06);

    padding: 20px;

    border-radius: 20px;

    margin-bottom: 18px;

    backdrop-filter: blur(8px);
}

.metric-card {

    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.15),
        rgba(139,92,246,0.10)
    );

    padding: 20px;

    border-radius: 20px;

    border: 1px solid rgba(255,255,255,0.08);
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
# SESSION STATE
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

        with open(
            'sentiment_model.pkl',
            'rb'
        ) as f:

            model = pickle.load(f)

        with open(
            'tfidf_vectorizer.pkl',
            'rb'
        ) as f:

            vectorizer = pickle.load(f)

    except Exception as e:

        st.error("Model files not found.")

        st.code(str(e))

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

    st.subheader("🎤 Voice Input")

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

        user_input = user_input.strip()

        if not user_input:

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
                ).translate(user_input)

            except:

                translated_text = user_input

            st.subheader("🌍 Translated Text")

            st.success(translated_text)

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

            label = str(prediction).upper()

            # =========================================================
            # DATABASE SAVE
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

            st.subheader("📌 Prediction")

            if label == "POSITIVE":

                st.success(f"{label} 😊")

            elif label == "NEGATIVE":

                st.error(f"{label} 😔")

            else:

                st.info(f"{label} 😐")

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
                        rgba(59,130,246,0.15),
                        rgba(139,92,246,0.12)
                    );
                    padding: 35px;
                    border-radius: 25px;
                    border: 1px solid rgba(255,255,255,0.08);
                    backdrop-filter: blur(10px);
                    text-align: center;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
                ">

                    <div style="
                        font-size:75px;
                        font-weight:800;
                        color:{confidence_color};
                        margin-bottom:10px;
                    ">
                        {confidence:.2f}%
                    </div>

                    <div style="
                        width:100%;
                        height:22px;
                        background:#111827;
                        border-radius:30px;
                        overflow:hidden;
                        margin-top:20px;
                        border:1px solid rgba(255,255,255,0.06);
                    ">

                        <div style="
                            width:{confidence}%;
                            height:100%;
                            border-radius:30px;
                            background: linear-gradient(
                                90deg,
                                #3B82F6,
                                #8B5CF6,
                                #22C55E
                            );
                        ">
                        </div>

                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-top:10px;
                        color:#94A3B8;
                        font-size:14px;
                    ">
                        <span>Low Confidence</span>
                        <span>High Confidence</span>
                    </div>

                    <p style="
                        margin-top:18px;
                        color:#E2E8F0;
                        font-size:18px;
                        font-weight:500;
                    ">
                        AI Prediction Confidence
                    </p>

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

                    st.pyplot(fig2)

            except Exception as e:

                st.warning(
                    "Emotion detection currently unavailable"
                )

                st.code(str(e))

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
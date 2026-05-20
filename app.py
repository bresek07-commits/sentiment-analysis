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

.stTextInput input {
    background-color: #1E293B !important;
    color: white !important;
    border-radius: 12px !important;
    border: 2px solid #334155 !important;
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

    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:20px;
            margin-bottom:25px;
        ">

            <div style="
                font-size:72px;
                line-height:1;
            ">
                🔐
            </div>

            <div style="
                font-size:58px;
                font-weight:800;
                background:linear-gradient(
                    90deg,
                    #F97316,
                    #FB923C,
                    #FDBA74
                );
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                line-height:1.1;
            ">
                Login
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

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

    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:20px;
            margin-bottom:25px;
        ">

            <div style="
                font-size:72px;
                line-height:1;
            ">
                😊
            </div>

            <div style="
                font-size:58px;
                font-weight:800;
                background:linear-gradient(
                    90deg,
                    #F97316,
                    #FB923C,
                    #FDBA74
                );
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                line-height:1.1;
            ">
                Advanced Sentiment <br>
                Analysis AI
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "AI-powered multilingual sentiment analysis"
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
                    """
                    <div class="prediction-positive">
                        POSITIVE 😊
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif label == "NEGATIVE":

                st.markdown(
                    """
                    <div class="prediction-negative">
                        NEGATIVE 😔
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="prediction-neutral">
                        NEUTRAL 😐
                    </div>
                    """,
                    unsafe_allow_html=True
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
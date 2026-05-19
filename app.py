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
# MODERN UI CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
MAIN APP
========================================================= */

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #020617
    );
    color: white;
}

/* =========================================================
HEADINGS
========================================================= */

h1 {
    font-size: 58px !important;
    font-weight: 800 !important;
    color: white !important;
    text-align: center;
    margin-bottom: 10px;
}

h2, h3 {
    color: white !important;
    font-weight: 700 !important;
}

/* =========================================================
TEXT
========================================================= */

p, label {
    color: #d1d5db !important;
    font-size: 17px !important;
}

/* =========================================================
TEXT AREA
========================================================= */

textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 16px !important;
    border: 1px solid #334155 !important;
    padding: 18px !important;
    font-size: 18px !important;
}

/* =========================================================
BUTTONS
========================================================= */

.stButton button {

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 14px !important;

    height: 52px !important;

    width: 170px !important;

    font-size: 18px !important;

    font-weight: 700 !important;

    transition: 0.3s ease-in-out !important;

    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.stButton button:hover {

    transform: scale(1.03);

    box-shadow: 0 6px 25px rgba(59,130,246,0.4);
}

/* =========================================================
CARDS
========================================================= */

.history-card {

    background: rgba(30,41,59,0.7);

    backdrop-filter: blur(12px);

    padding: 25px;

    border-radius: 22px;

    margin-bottom: 20px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background: #111827;
}

/* =========================================================
METRIC
========================================================= */

[data-testid="metric-container"] {

    background-color: #1e293b;

    border: 1px solid #334155;

    padding: 20px;

    border-radius: 18px;
}

/* =========================================================
EXPANDER
========================================================= */

.streamlit-expanderHeader {

    font-size: 22px !important;

    font-weight: bold !important;

    color: white !important;
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
# LOGIN
# =========================================================

if not st.session_state.logged_in:

    st.markdown("""
    <h1>🔐 Login</h1>
    """, unsafe_allow_html=True)

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

    st.markdown("""
    <h1>
    ✨ Advanced Sentiment Analysis AI
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-bottom:30px;'>

    <h4 style='color:#cbd5e1;'>

    AI-powered multilingual sentiment and emotion analysis

    </h4>

    <p style='color:#94a3b8;'>

    Supports English, Hindi, Urdu and more 🌍

    </p>

    </div>
    """, unsafe_allow_html=True)

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

    if "text_value" not in st.session_state:
        st.session_state.text_value = ""

    if voice_text:
        st.session_state.text_value = voice_text

        st.success(
            "Voice recognized!"
        )

    # =========================================================
    # TEXT AREA
    # =========================================================

    user_input = st.text_area(
        "Enter text",
        value=st.session_state.text_value,
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
            # SAVE DATABASE
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
            # RESULT
            # =========================================================

            st.subheader(
                "📌 Prediction"
            )

            if label == "POSITIVE":

                st.success(
                    f"POSITIVE 😊"
                )

            else:

                st.error(
                    f"NEGATIVE 😔"
                )

            # =========================================================
            # CONFIDENCE SCORE
            # =========================================================

            st.subheader(
                "📈 Confidence Score"
            )

            fig, ax = plt.subplots(
                figsize=(8, 2)
            )

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)

            ax.axis('off')

            colors = [
                "#ff0000",
                "#ff6600",
                "#ffcc00",
                "#ccff00",
                "#33cc33"
            ]

            ranges = [
                (0, 20),
                (20, 40),
                (40, 60),
                (60, 80),
                (80, 100)
            ]

            for i, r in enumerate(ranges):

                ax.barh(
                    0.5,
                    r[1] - r[0],
                    left=r[0],
                    height=0.3,
                    color=colors[i]
                )

            ax.plot(
                [confidence, confidence],
                [0.3, 0.8],
                color='black',
                linewidth=5
            )

            ax.text(
                confidence,
                0.88,
                f"{confidence:.1f}%",
                ha='center',
                fontsize=18,
                fontweight='bold'
            )

            ax.text(
                0,
                0.1,
                "LOW",
                fontsize=14
            )

            ax.text(
                100,
                0.1,
                "HIGH",
                fontsize=14,
                ha='right'
            )

            plt.title(
                "CONFIDENCE",
                fontsize=24,
                fontweight='bold'
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

                        emotion_names.append(k.capitalize())
                        emotion_values.append(v)

                if len(emotion_names) == 0:

                    st.info(
                        "No strong emotions detected"
                    )

                else:

                    fig2, ax2 = plt.subplots(
                        figsize=(8, 8)
                    )

                    colors = [
                        "#FFD166",
                        "#EF476F",
                        "#06D6A0",
                        "#118AB2",
                        "#9B5DE5"
                    ]

                    wedges, texts = ax2.pie(
                        emotion_values,
                        labels=emotion_names,
                        startangle=90,
                        colors=colors[:len(emotion_names)],
                        wedgeprops=dict(
                            width=0.35,
                            edgecolor='white'
                        )
                    )

                    centre_circle = plt.Circle(
                        (0, 0),
                        0.45,
                        fc='white'
                    )

                    fig2.gca().add_artist(
                        centre_circle
                    )

                    plt.title(
                        "The Feelings Wheel",
                        fontsize=28,
                        fontweight='bold'
                    )

                    ax2.axis('equal')

                    st.pyplot(
                        fig2
                    )

                    st.markdown(
                        "### 🎭 Emotion Breakdown"
                    )

                    for name, value in zip(
                        emotion_names,
                        emotion_values
                    ):

                        st.progress(
                            float(value)
                        )

                        st.write(
                            f"**{name}** — {value*100:.0f}%"
                        )

            except Exception as e:

                st.warning(
                    "Emotion detection currently unavailable"
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

            else:

                st.write(
                    "That's great to hear! Keep smiling 😀"
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
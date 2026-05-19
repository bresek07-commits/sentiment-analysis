# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import sqlite3
from deep_translator import GoogleTranslator
import text2emotion as te
from streamlit_mic_recorder import speech_to_text
import nltk
import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------------------------------
# DOWNLOAD NLTK DATA
# -------------------------------------------------

nltk.download('punkt')

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Advanced Sentiment Analysis AI",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    h1 {
        font-size: 52px !important;
        font-weight: bold;
        color: white;
    }

    h2, h3 {
        color: white;
    }

    .stButton button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
    }

    .stButton button:hover {
        background-color: #1D4ED8;
    }

    textarea {
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# DATABASE
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

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------

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
    # TITLE
    # -------------------------------------------------

    st.title("🧠 Advanced Sentiment Analysis AI")

    st.write(
        "AI-powered multilingual sentiment and emotion analysis"
    )

    st.write(
        "Supports English, Hindi, Urdu and more"
    )

    # -------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------

    classifier = pipeline(
        "sentiment-analysis"
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

            st.warning("Please enter some text")

        else:

            # -------------------------------------------------
            # TRANSLATE
            # -------------------------------------------------

            translated_text = GoogleTranslator(
                source='auto',
                target='en'
            ).translate(
                user_input
            )

            st.subheader("🌍 Translated Text")

            st.info(translated_text)

            # -------------------------------------------------
            # SENTIMENT ANALYSIS
            # -------------------------------------------------

            result = classifier(
                translated_text
            )

            label = result[0]['label']

            confidence = result[0]['score']

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
            # RESULT
            # -------------------------------------------------

            st.subheader("📊 Sentiment Result")

            if label == "POSITIVE":

                st.success(
                    f"Positive 😀 ({round(confidence * 100, 2)}%)"
                )

            else:

                st.error(
                    f"Negative 😔 ({round(confidence * 100, 2)}%)"
                )

            # -------------------------------------------------
            # AI RESPONSE
            # -------------------------------------------------

            st.subheader("🤖 AI Assistant")

            if label == "NEGATIVE":

                st.warning(
                    "I'm sorry you're feeling upset. Hope things improve soon ❤️"
                )

            else:

                st.success(
                    "That's great to hear! Keep smiling 😀"
                )

            # -------------------------------------------------
            # PROFESSIONAL CONFIDENCE CHART
            # -------------------------------------------------

            st.subheader("📈 Confidence Score")

            positive_score = confidence * 100

            negative_score = 100 - positive_score

            chart_labels = [
                "Positive",
                "Negative"
            ]

            if label == "NEGATIVE":

                chart_values = [
                    negative_score,
                    positive_score
                ]

            else:

                chart_values = [
                    positive_score,
                    negative_score
                ]

            fig1, ax1 = plt.subplots(
                figsize=(4, 4)
            )

            ax1.pie(
                chart_values,
                labels=chart_labels,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={
                    'width': 0.4
                }
            )

            ax1.axis('equal')

            st.pyplot(fig1)

            # -------------------------------------------------
            # EMOTION DETECTION
            # -------------------------------------------------

            st.subheader("😊 Emotion Detection")

            try:

                emotions = te.get_emotion(
                    translated_text
                )

                emotions = {
                    k: v for k, v in emotions.items()
                    if v > 0
                }

                if len(emotions) == 0:

                    st.info(
                        "No strong emotion detected"
                    )

                else:

                    # -------------------------------------------------
                    # FEELINGS WHEEL STYLE CHART
                    # -------------------------------------------------

                    fig, ax = plt.subplots(
                        figsize=(7, 7),
                        subplot_kw=dict(polar=True)
                    )

                    emotion_labels = list(
                        emotions.keys()
                    )

                    emotion_values = list(
                        emotions.values()
                    )

                    total = sum(
                        emotion_values
                    )

                    sizes = [
                        (v / total) * 2 * np.pi
                        for v in emotion_values
                    ]

                    angles = np.cumsum(
                        [0] + sizes[:-1]
                    )

                    colors = [
                        "#FFD166",
                        "#EF476F",
                        "#06D6A0",
                        "#118AB2",
                        "#9B5DE5"
                    ]

                    ax.bar(
                        angles,
                        emotion_values,
                        width=sizes,
                        bottom=2,
                        color=colors[:len(emotion_values)],
                        edgecolor="white",
                        linewidth=2,
                        align='edge'
                    )

                    # -------------------------------------------------
                    # LABELS
                    # -------------------------------------------------

                    for angle, label_name, value in zip(
                        angles,
                        emotion_labels,
                        emotion_values
                    ):

                        ax.text(
                            angle + 0.2,
                            3.2,
                            f"{label_name}\n{round(value*100)}%",
                            ha='center',
                            va='center',
                            fontsize=11,
                            fontweight='bold',
                            color='white'
                        )

                    # -------------------------------------------------
                    # STYLE
                    # -------------------------------------------------

                    ax.set_theta_offset(
                        np.pi / 2
                    )

                    ax.set_theta_direction(-1)

                    ax.set_yticklabels([])

                    ax.set_xticklabels([])

                    ax.grid(False)

                    ax.spines['polar'].set_visible(False)

                    fig.patch.set_facecolor("#0E1117")

                    ax.set_facecolor("#0E1117")

                    st.pyplot(fig)

            except Exception as e:

                st.warning(
                    "Emotion detection currently unavailable"
                )

                st.code(
                    str(e)
                )

    # -------------------------------------------------
    # PREDICTION HISTORY
    # -------------------------------------------------

    with st.expander(
        "📜 Prediction History",
        expanded=False
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

                text = row[0]

                sentiment = row[1]

                confidence = row[2]

                created_at = row[3]

                if sentiment == "POSITIVE":

                    sentiment_color = "#22C55E"

                else:

                    sentiment_color = "#EF4444"

                st.markdown(
                    f"""
                    <div style="
                        background-color:#111827;
                        padding:20px;
                        border-radius:15px;
                        margin-bottom:15px;
                        border-left:6px solid {sentiment_color};
                    ">

                    <h4 style="color:white;">
                    📝 {text}
                    </h4>

                    <p style="color:#D1D5DB;">
                    📌 Sentiment:
                    <span style="color:{sentiment_color}; font-weight:bold;">
                    {sentiment}
                    </span>
                    </p>

                    <p style="color:#D1D5DB;">
                    📊 Confidence:
                    {round(confidence * 100, 2)}%
                    </p>

                    <p style="color:#9CA3AF;">
                    ⏰ {created_at}
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # -------------------------------------------------
    # ADMIN DASHBOARD
    # -------------------------------------------------

    if username == "admin":

        st.subheader("📊 Admin Dashboard")

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

        admin_df = pd.DataFrame(
            rows,
            columns=[
                "Username",
                "Text",
                "Sentiment",
                "Confidence",
                "Time"
            ]
        )

        st.dataframe(
            admin_df,
            use_container_width=True
        )
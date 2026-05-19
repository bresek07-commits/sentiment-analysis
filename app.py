# =========================================================
# ADVANCED SENTIMENT ANALYSIS AI
# =========================================================

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
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
    page_title="Advanced Sentiment AI",
    page_icon="😊",
    layout="centered"
)

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
# DATABASE
# =========================================================

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

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.success(
        f"Welcome {name}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.name = ""

        st.rerun()

    # =====================================================
    # TITLE
    # =====================================================

    st.title(
        "😊 Advanced Sentiment Analysis AI"
    )

    st.markdown(
        """
        ### AI-powered multilingual sentiment and emotion analysis

        Supports English, Hindi, Urdu and more
        """
    )

    # =====================================================
    # VOICE INPUT
    # =====================================================

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

        st.write(
            voice_text
        )

    # =====================================================
    # TEXT INPUT
    # =====================================================

    user_input = st.text_area(
        "Enter text",
        value=voice_text if voice_text else "",
        height=180
    )

    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button("Analyze"):

        if user_input.strip() == "":

            st.warning(
                "Please enter some text"
            )

        else:

            # =============================================
            # TRANSLATE TEXT
            # =============================================

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

            # =============================================
            # SENTIMENT PREDICTION
            # =============================================

            transformed_text = vectorizer.transform(
                [translated_text]
            )

            prediction = model.predict(
                transformed_text
            )[0]

            probabilities = model.predict_proba(
                transformed_text
            )

            confidence = np.max(
                probabilities
            )

            label = "POSITIVE"

            if prediction == 0:

                label = "NEGATIVE"

            # =============================================
            # RESULT
            # =============================================

            st.subheader(
                "🎯 Sentiment Result"
            )

            if label == "POSITIVE":

                st.success(
                    f"Prediction: {label}"
                )

            else:

                st.error(
                    f"Prediction: {label}"
                )

            # =============================================
            # CONFIDENCE GAUGE
            # =============================================

            st.subheader(
                "📈 Confidence Score"
            )

            confidence_percent = confidence * 100

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.axis('off')

            theta = np.linspace(
                np.pi,
                2 * np.pi,
                100
            )

            ax.plot(
                np.cos(theta),
                np.sin(theta),
                linewidth=35
            )

            angle = np.pi + (
                confidence_percent / 100
            ) * np.pi

            ax.arrow(
                0,
                0,
                0.7 * np.cos(angle),
                0.7 * np.sin(angle),
                width=0.03
            )

            ax.text(
                0,
                1.2,
                "CONFIDENCE",
                ha='center',
                fontsize=24,
                fontweight='bold'
            )

            ax.text(
                -1.1,
                -0.1,
                "LOW",
                fontsize=16
            )

            ax.text(
                0.9,
                -0.1,
                "HIGH",
                fontsize=16
            )

            ax.text(
                0,
                -0.35,
                f"{confidence_percent:.1f}%",
                ha='center',
                fontsize=22,
                fontweight='bold'
            )

            ax.set_xlim(-1.2, 1.2)

            ax.set_ylim(-1.2, 1.4)

            st.pyplot(fig)

            # =============================================
            # EMOTION DETECTION
            # =============================================

            st.subheader(
                "😊 Emotion Detection"
            )

            try:

                emotions = te.get_emotion(
                    translated_text
                )

                emotion_df = pd.DataFrame({
                    "Emotion": list(emotions.keys()),
                    "Score": list(emotions.values())
                })

                st.dataframe(
                    emotion_df,
                    use_container_width=True
                )

                fig2, ax2 = plt.subplots(
                    figsize=(7, 7)
                )

                values = list(
                    emotions.values()
                )

                labels = list(
                    emotions.keys()
                )

                colors = [
                    '#FF9999',
                    '#66B3FF',
                    '#99FF99',
                    '#FFCC99',
                    '#C2C2F0'
                ]

                ax2.pie(
                    values,
                    labels=labels,
                    colors=colors,
                    startangle=90,
                    wedgeprops={
                        'width': 0.4,
                        'edgecolor': 'white'
                    },
                    autopct='%1.1f%%'
                )

                centre_circle = plt.Circle(
                    (0, 0),
                    0.55,
                    fc='white'
                )

                fig2.gca().add_artist(
                    centre_circle
                )

                ax2.set_title(
                    "Emotion Wheel",
                    fontsize=24,
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

            # =============================================
            # AI ASSISTANT
            # =============================================

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

            # =============================================
            # SAVE TO DATABASE
            # =============================================

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
                    float(confidence)
                )
            )

            conn.commit()

    # =====================================================
    # HISTORY
    # =====================================================

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

                text = row[0]

                sentiment = row[1]

                confidence_value = row[2]

                created_at = row[3]

                st.markdown(
                    f"""
                    <div style="
                        background-color:#112240;
                        padding:20px;
                        border-radius:15px;
                        margin-bottom:15px;
                    ">

                    <h4>📝 Text</h4>
                    <p>{text}</p>

                    <h4>➡️ Sentiment</h4>
                    <p>{sentiment}</p>

                    <h4>📊 Confidence</h4>
                    <p>{confidence_value:.2%}</p>

                    <h4>⏰ Time</h4>
                    <p>{created_at}</p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =====================================================
    # ADMIN DASHBOARD
    # =====================================================

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

        st.dataframe(
            rows,
            use_container_width=True
        )
import streamlit as st
import pickle

import joblib
model = joblib.load("logreg_model.pkl")
tfidf = joblib.load("tfidf.pkl")


st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #5D6D7E;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #D6EAF8;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #154360;
        margin-top: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎭 Emotion Detection App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enter a sentence and I’ll try to detect the emotion behind it 🤖</div>", unsafe_allow_html=True)

user_input = st.text_area("✍️ Type your text here:", height=150, placeholder="Example: I am so excited for the trip!")

emojis = {
    "joy": "😃",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️"
}

label_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        vec = tfidf.transform([user_input])
        prediction = model.predict(vec)[0]
        prediction = int(prediction)
        emotion = label_map.get(prediction, "unknown")
        emoji = emojis.get(emotion.lower(), "🤔")
        st.markdown(f"<div class='result-box'>Predicted Emotion: {emoji} {emotion}</div>", unsafe_allow_html=True)

st.sidebar.header("About this App")
st.sidebar.info("This app uses **Logistic Regression + TF-IDF** to detect emotions from text.  Accuracy: ~85% 🎯")

st.sidebar.header("How it Works?")
st.sidebar.write("1. Text is converted into numerical features using **TF-IDF**.")
st.sidebar.write("2. Logistic Regression classifies the emotion.")

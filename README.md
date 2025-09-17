# 🎭 Emotion Detection App  

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](https://emotiondetect-2ysbluimrqdsuvqh87qbz4.streamlit.app/)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)  
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange?logo=scikitlearn)](https://scikit-learn.org/)

**Live app:** [🔗 Open Live App (Streamlit)](https://emotiondetect-2ysbluimrqdsuvqh87qbz4.streamlit.app/)

## 📖 Overview  

The **Emotion Detection App** is an **NLP-powered web application** that predicts the underlying emotion in user input text.  
It is built using **TF-IDF + Logistic Regression**, achieving **85% accuracy 🎯**.   

## ✨ Features :

🎨 Clean & modern Streamlit UI
😃 Six supported emotions: Joy, Sadness, Anger, Fear, Surprise, Love
🤖 Emoji-based prediction for better visualization
☁️ Deployed on Streamlit Community Cloud


⚙️ Installation & Usage
# Clone the repository
git clone https://github.com/YOUR_USERNAME/EmotionDetect.git
cd EmotionDetect

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py


🧠 How It Works
1. Input text is preprocessed and converted into vectors using TF-IDF
2. Logistic Regression classifies the text into one of six emotions
3. The app displays the predicted emotion with an emoji

🚀 Deployment
This project is deployed on Streamlit Community Cloud.
👉 https://emotiondetect-2ysbluimrqdsuvqh87qbz4.streamlit.app/

📌 Future Improvements
- Add more emotions & larger datasets
- Deploy with Docker / Hugging Face Spaces
- Use Transformers (BERT, RoBERTa) for higher accuracy

👨‍💻 Author
Made by Krish (https://github.com/Krish00711)


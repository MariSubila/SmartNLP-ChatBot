# 🤖 SmartNLP Chatbot

SmartNLP Chatbot is an AI-powered conversational assistant built using **Streamlit**, **NLTK**, and **scikit-learn**. It allows users to interact with a trained chatbot using natural language. The app supports both **text input** and (optionally) **voice input**.

---

## 🧠 Features

- Predefined intents-based chatbot using NLP techniques
- Interactive chat interface with **Streamlit**
- Text classification and prediction using **scikit-learn**
- Intent matching with confidence score
- Lightweight and easy to run locally

---

## 🖥️ How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MariSubila/smartnlp-chatbot.git
   cd smartnlp-chatbot
---

## Create and activate a virtual environment (optional but recommended):
python -m venv venv
venv\Scripts\activate   # On Windows

---
## Install dependencies:
pip install -r requirements.txt

---
## Run the Streamlit app:
streamlit run app.py

---
## Note
⚡ Voice input (via microphone) requires pyaudio.

🔊 However, since pyaudio is not installed on my laptop and not supported by some systems, the app will only support text-based chat input in such cases.

---
## 🛠️ Built With
* Python

* Streamlit

* NLTK

* scikit-learn

* TextBlob

---
## Project Structure

smartnlp-chatbot/
│
├── intents.json           # Predefined intents for the chatbot
├── app.py                 # Streamlit app file
├── train_chatbot.py       # Training script using NLTK and sklearn
├── chatbot_model.pkl      # Trained model
├── vectorizer.pkl         # Text vectorizer
├── requirements.txt       # List of dependencies
└── README.md              # This file

## 🚀 Future Enhancements
-Voice input via microphone (when pyaudio is supported)

-Chat history and logging

-Improved intent matching with deep learning



import streamlit as st
import json
import random
import pickle
import speech_recognition as sr
import pyttsx3
import threading

# Load model and data
model = pickle.load(open("model/chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("model/words.pkl", "rb"))
label_encoder = pickle.load(open("model/classes.pkl", "rb"))
intents = json.load(open("data/intents.json"))

# Voice output
def speak(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

# Voice input
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening...")
        audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio)
            st.success(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")
        return ""

# Predict intent
def predict_class(user_input):
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]
    intent = label_encoder.inverse_transform([prediction])[0]
    return intent

# Get response
def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Streamlit UI
st.title("🧠 SmartNLP ChatBot")
st.markdown("Ask me anything!")

chat_mode = st.radio("Choose Input Method:", ["Text", "Voice 🎤"])

if chat_mode == "Text":
    user_input = st.text_input("💬 Type your message:")
    if st.button("Send"):
        if user_input:
            intent = predict_class(user_input)
            response = get_response(intent, intents)
            st.success(f"🤖 {response}")
            speak(response)
        else:
            st.warning("Please type a message.")

else:
    if st.button("🎙️ Speak"):
        user_input = get_voice_input()
        if user_input:
            intent = predict_class(user_input)
            response = get_response(intent, intents)
            st.success(f"🤖 {response}")
            speak(response)

import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])

def load_intents(path='intents.json'):
    with open(path, 'r') as f:
        return json.load(f)

def get_response(user_input, model, vectorizer, intents):
    processed = preprocess_text(user_input)
    vec = vectorizer.transform([processed])
    intent = model.predict(vec)[0]

    for item in intents['intents']:
        if item['tag'] == intent:
            return random.choice(item['responses'])
    return "Sorry, I didn’t understand that."

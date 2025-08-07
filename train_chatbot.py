import json
import random
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Load intents
with open('data/intents.json') as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()

corpus = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern.lower())
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
        corpus.append(' '.join(lemmas))
        labels.append(intent['tag'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

model = MultinomialNB()
model.fit(X, y)

# Save model and encoders
with open("model/chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/words.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model/classes.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Training complete. Model saved.")

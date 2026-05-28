import json
import tensorflow as tk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time

def load_tokenizer():
    with open('artifacts/tokenizer_v1.json') as file:
        tokenizer_data = json.load(file)

    tokenizer = tokenizer_from_json(tokenizer_data)
    return tokenizer

def load_selected_model():
    model = load_model('artifacts/model_v1.keras')
    return model

def get_top_k_predictions(model, user_input, k):
    suggestions = []

    tokenizer = load_tokenizer()
    model = load_selected_model()
    tokenized_text = tokenizer.texts_to_sequences([user_input])[0]

    padded_token_text = pad_sequences([tokenized_text], maxlen = 29, padding = 'pre')

    top_k_indices  = np.argsort(model.predict([padded_token_text])[0])[-k:][::-1]

    for i in top_k_indices:
        for word, index in tokenizer.word_index.items():
            if index == i:
                suggestions.append(word)
    
    return suggestions





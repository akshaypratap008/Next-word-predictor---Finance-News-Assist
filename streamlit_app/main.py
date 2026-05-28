import streamlit as st
import json
from tensorflow.keras.models import load_model      
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import requests

from utils import load_selected_model, load_tokenizer, get_top_k_predictions

# API
PREDICTION_API_URL = 'http://127.0.0.1:8000/predict'

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

if "predict_mode" not in st.session_state:
    st.session_state.predict_mode = False

st.set_page_config(page_title="Next Word Predictor", layout="wide")


# sidebar
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["LSTM_v1", "GRU_v1", "BiLSTM_v1"],
)

num_words_to_generate = st.sidebar.slider(
    "Number of words to auto‑generate",
    min_value=1,
    max_value=20,
    value=5,
)

top_k = st.sidebar.slider(
    "Top‑k Suggestions",
    min_value=1,
    max_value=5,
    value=3,
)

if st.sidebar.button("Reset Text"):
    st.session_state.user_text = ""
    st.session_state.predict_mode = False
    st.experimental_rerun()




# prediction function 
def run_prediction():
        user_input = st.session_state.user_text
        
        # api request
        payload = {
            'user_input': user_input,
            'k': top_k
        }

        response = requests.post(url = PREDICTION_API_URL, json=payload)
        top_words = response.json()['suggestions']

        st.subheader("🔮 Model Suggestions")
        cols = st.columns(top_k)

        # top k suggestions
        for i, col in enumerate(cols):
            with col:
                with st.container():
                    st.info(top_words[i])


        # Auto-generate continuation
        st.subheader("📝 Auto‑Generated Continuation")
        placeholder = st.empty()
        progress = st.progress(0)

        generated_text = user_input

        for i in range(num_words_to_generate):
            payload = {
            'user_input': generated_text,
            'k': num_words_to_generate
            }
            response = requests.post(url = PREDICTION_API_URL, json=payload)
            top_words = response.json()['suggestions']

            next_word = top_words[0]
            generated_text += " " + next_word

            placeholder.markdown(f"**Generated so far:** {generated_text}")
            progress.progress((i+1) / num_words_to_generate)

# main UI
st.title("🧠 Next Word Predictor App")
st.write("Interactive next‑word prediction with top‑k suggestions.")

st.subheader("✍️ Type your text")

st.text_area(
    "Start typing...",
    key="user_text",
    height=150,
)

# Predict button
if st.button("Predict Next Word"):
    st.session_state.predict_mode = True

# Run prediction mode
if st.session_state.predict_mode and st.session_state.user_text.strip():
    run_prediction()
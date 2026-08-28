import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

st.set_page_config(
    page_title="NeuroChat AI",
    page_icon="🧠",
    layout="centered",
)

# ---------- Simple Dark UI ----------
st.markdown(
    """
    <style>
    .stApp {
        background: #080b12;
        color: #f8fafc;
    }

    .block-container {
        max-width: 850px;
        padding-top: 3rem;
    }

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 35px;
    }

    .result {
        background: #111827;
        border: 1px solid #293241;
        border-radius: 14px;
        padding: 20px;
        margin-top: 20px;
        font-size: 20px;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🧠 NeuroChat AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">LSTM Powered Next Word Prediction</div>',
    unsafe_allow_html=True,
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/next_word_lstm_model.keras",
        compile=False,
    )

@st.cache_resource
def load_tokenizer():
    with open("model/tokenizer.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error("❌ Model or tokenizer could not be loaded.")
    st.code(str(e))
    st.stop()

input_length = model.input_shape[1]

# ---------- Prediction ----------
def predict_next_words(text, num_words=1):
    current_text = text.strip()

    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([current_text])[0]

        if not sequence:
            return None

        sequence = sequence[-input_length:]

        padded = pad_sequences(
            [sequence],
            maxlen=input_length,
            padding="pre",
        )

        probabilities = model.predict(padded, verbose=0)[0]
        next_index = int(np.argmax(probabilities))

        next_word = tokenizer.index_word.get(next_index)

        if not next_word or next_word == "<UNK>":
            return None

        current_text = f"{current_text} {next_word}".strip()

    return current_text

# ---------- Controls ----------
num_words = st.selectbox(
    "Number of words to predict",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: f"{x} word" if x == 1 else f"{x} words",
)

text = st.text_input(
    "Enter your sentence",
    placeholder="Example: machine learning is",
)

if st.button("🔮 Predict Next Words", use_container_width=True, key="predict_button"):
    if not text.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Predicting..."):
            result = predict_next_words(text, num_words)

        if result is None:
            st.warning("Could not predict the next word. Try another sentence.")
        else:
            st.markdown(
                f'<div class="result">{result}</div>',
                unsafe_allow_html=True,
            )

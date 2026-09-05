import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

st.set_page_config(
    page_title="NextWord AI",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #080b12;
    color: #f8fafc;
}

.block-container {
    max-width: 850px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 6px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 32px;
}

.prediction-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 18px;
}

.word-box {
    background: #111827;
    border: 1px solid #293241;
    border-radius: 12px;
    padding: 15px 20px;
    margin: 10px 0;
    transition: 0.2s ease;
}

.word-box:hover {
    border-color: #475569;
}

.word {
    font-size: 18px;
    font-weight: 600;
    color: #f8fafc;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 45px;
    padding-top: 18px;
    border-top: 1px solid #1e293b;
}

[data-testid="stSidebar"] {
    background: #0d111a;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    height: 45px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🧠 NextWord AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">LSTM Powered Next Word Prediction</div>',
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/next_word_lstm_model.keras",
        compile=False
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


with st.sidebar:
    st.header("⚙️ Settings")

    num_words = st.slider(
        "Number of words to predict",
        min_value=1,
        max_value=5,
        value=1
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Architecture:** LSTM")
    st.write("**LSTM Units:** 150")
    st.write("**Embedding Size:** 100")
    st.write("**Vocabulary:** ~4,994")
    st.write(f"**Input Length:** {input_length}")

    st.markdown("---")

    st.caption(
    "Custom-trained LSTM for intelligent next word prediction."
    )


def predict_next_words(text, num_words):
    current_text = text.strip()
    predictions = []

    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([current_text])[0]

        if not sequence:
            return None

        sequence = sequence[-input_length:]

        padded = pad_sequences(
            [sequence],
            maxlen=input_length,
            padding="pre"
        )

        probabilities = model.predict(
            padded,
            verbose=0
        )[0]

        next_index = int(np.argmax(probabilities))
        confidence = float(probabilities[next_index])

        next_word = tokenizer.index_word.get(next_index)

        if not next_word or next_word == "<UNK>":
            return None

        predictions.append({
            "word": next_word,
            "confidence": confidence
        })

        current_text = f"{current_text} {next_word}".strip()

    return predictions


text = st.text_input(
    "Enter your sentence",
    placeholder="Example: machine learning is"
)


if st.button(
    "🔮 Predict Next Words",
    use_container_width=True
):
    if not text.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Predicting..."):
            predictions = predict_next_words(
                text,
                num_words
            )

        if predictions is None:
            st.warning(
                "Could not predict the next word. "
                "Try another sentence."
            )
        else:
            st.markdown(
                '<div class="prediction-title">✨ Next Word Predictions</div>',
                unsafe_allow_html=True
            )

            for index, prediction in enumerate(predictions, 1):
                word = prediction["word"]
                percentage = prediction["confidence"] * 100

                st.markdown(
                    f"""
                    <div class="word-box">
                        <span class="word">
                            {index}. {word} — {percentage:.2f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


st.markdown(
    """
    <div class="footer">
        NextWord AI • Custom-trained LSTM model for next word prediction.<br>
        Built with Streamlit and TensorFlow.<br>
        © 2026 by Subham Das
    </div>
    """,
    unsafe_allow_html=True
)
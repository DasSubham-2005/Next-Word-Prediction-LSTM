import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


st.set_page_config(
    page_title="NeuroChat AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(99,102,241,.18), transparent 30%),
            radial-gradient(circle at 100% 100%, rgba(168,85,247,.14), transparent 35%),
            #080b12;
        color: #f8fafc;
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .neuro-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .neuro-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 28px;
    }

    .prediction-card {
        background: rgba(30,41,59,.78);
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 18px;
        padding: 20px;
        margin: 14px 0;
    }

    .prediction-chip {
        display: inline-block;
        padding: 8px 13px;
        margin: 5px 4px;
        border-radius: 18px;
        background: rgba(99,102,241,.14);
        border: 1px solid rgba(129,140,248,.30);
        color: #c7d2fe;
        font-weight: 600;
        font-size: 14px;
    }

    .section-label {
        color: #cbd5e1;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .best-label {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 12px;
    }

    .welcome {
        text-align: center;
        padding: 55px 20px;
        color: #94a3b8;
    }

    .welcome strong {
        color: #c7d2fe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div style="text-align:center;font-size:42px;">🧠</div>', unsafe_allow_html=True)
st.markdown('<div class="neuro-title">NeuroChat AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="neuro-subtitle">LSTM Powered Next Word Prediction</div>',
    unsafe_allow_html=True,
)


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
    st.info(
        "Make sure these files exist: "
        "model/next_word_lstm_model.keras and model/tokenizer.pkl"
    )
    st.stop()


# MODEL INFORMATION
# Based on the uploaded notebook:
# vocab_size = 4994
# embedding_dim = 100
# LSTM units = 150
# optimizer = Adam
# loss = categorical_crossentropy

vocab_size = len(tokenizer.word_index) + 1
input_length = model.input_shape[1]


with st.sidebar:
    st.markdown("## ⚙️ Settings")

    top_k = st.slider(
        "Number of predictions",
        min_value=1,
        max_value=10,
        value=5,
    )

    temperature = st.slider(
        "Prediction creativity",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

    st.divider()

    st.markdown("### 🧬 Model")
    st.write("Architecture: **LSTM**")
    st.write("LSTM Units: **150**")
    st.write("Embedding: **100**")
    st.write(f"Vocabulary: **{vocab_size}**")
    st.write(f"Input length: **{input_length}**")

    st.divider()

    st.markdown("### 📊 Training")
    st.write("Optimizer: **Adam**")
    st.write("Loss: **Categorical Crossentropy**")
    st.write("Epochs: **100**")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# PREDICTION FUNCTION

def predict_next_words(text, top_k=5, temperature=1.0):
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return []

    # The notebook trains on x = padded_sequences[:, :-1].
    # Therefore the saved model expects input_length tokens.
    sequence = sequence[-input_length:]

    padded = pad_sequences(
        [sequence],
        maxlen=input_length,
        padding="pre",
    )

    probabilities = model.predict(padded, verbose=0)[0]

    # Temperature scaling
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.log(probabilities + 1e-10) / temperature
    probabilities = np.exp(probabilities - np.max(probabilities))
    probabilities = probabilities / np.sum(probabilities)

    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    results = []
    for index in top_indices:
        word = tokenizer.index_word.get(int(index), "<UNK>")
        confidence = float(probabilities[index]) * 100
        results.append((word, confidence))

    return results



# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

# QUICK TESTS
# Conversation/dialogue-style prompts

st.markdown("### 🧪 Quick Test")

test_prompts = [
    "how are you",
    "When are you",
    "I don't know",
    "the biggest news",
    "where are you",
    "Hello? Like he was",
    "do you want",
    "can you help",
]

cols = st.columns(2)

for i, test_text in enumerate(test_prompts):
    with cols[i % 2]:
        if st.button(test_text, key=f"test_{i}", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": test_text}
            )

            with st.spinner("🧠 LSTM is thinking..."):
                predictions = predict_next_words(
                    test_text,
                    top_k=top_k,
                    temperature=temperature,
                )

            st.session_state.messages.append(
                {"role": "assistant", "predictions": predictions}
            )
            st.rerun()

st.divider()


# CHAT HISTORY

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome">
            <div style="font-size:28px;">👋</div>
            <h3>Welcome to NeuroChat AI</h3>
            <p>
                Enter a conversation-style sentence and let the LSTM
                predict the most likely next words.
            </p>
            <p>
                Example: <strong>how are you</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])

    else:
        predictions = message["predictions"]

        with st.chat_message("assistant"):
            st.markdown("**🧠 Predicted Next Words**")

            if not predictions:
                st.warning(
                    "No known words were found. Try a sentence similar to "
                    "your training dataset."
                )
                continue

            # Native Streamlit display: no raw HTML tags in the prediction output.
            for word, confidence in predictions:
                st.markdown(
                    f"`{word}`  —  **{confidence:.2f}%**"
                )

            best_word, best_confidence = predictions[0]

            st.success(
                f"Best prediction: **{best_word}** "
                f"({best_confidence:.2f}%)"
            )


# CHAT INPUT

prompt = st.chat_input("Type something like: how are you...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.spinner("🧠 LSTM is thinking..."):
        predictions = predict_next_words(
            prompt,
            top_k=top_k,
            temperature=temperature,
        )

    st.session_state.messages.append(
        {"role": "assistant", "predictions": predictions}
    )

    st.rerun()

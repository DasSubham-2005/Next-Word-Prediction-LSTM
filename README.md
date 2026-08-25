# 🧠 NeuroChat AI — LSTM Next Word Prediction

NeuroChat AI is an LSTM-based Next Word Prediction project built with Python, TensorFlow/Keras, NumPy, and Streamlit.

The project takes a sequence of words as input and predicts the most probable next words using a trained LSTM language model. The trained model is integrated into a modern chatbot-style Streamlit interface.

---

## ✨ Features

- 🧠 LSTM-based Next Word Prediction
- 💬 Chatbot-style Streamlit interface
- 🔮 Top-K next-word predictions
- 📊 Prediction confidence
- ⚙️ Adjustable number of predictions
- 🌡️ Temperature control for focused or diverse predictions
- 🗑️ Clear chat functionality
- 🎨 Modern dark/neon interface
- ⚡ Fast prediction using a saved trained model
- 📦 Saved LSTM model and tokenizer
- 📈 Training accuracy and loss visualization
- 🧩 LSTM architecture visualization

---

## 🏗️ Model Architecture

The trained model follows this pipeline:

```text
Input Text
    ↓
Tokenizer
    ↓
Sequence Creation
    ↓
Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense Layer
    ↓
Softmax
    ↓
Next Word Prediction
```

### Model Configuration

| Component | Configuration |
|---|---|
| Model Type | LSTM |
| Embedding Dimension | 100 |
| LSTM Units | 150 |
| Vocabulary Size | 4994 |
| Output Layer | Dense |
| Output Activation | Softmax |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |

---

## 📁 Project Structure

```text
Next-Word-Prediction-LSTM/
│
├── app.py
├── README.md
├── requirements.txt
├── dataset.txt
├── .gitignore
│
└── model/
    ├── next_word_lstm_model.keras
    └── tokenizer.pkl
```

---

## 🔄 Project Workflow

### Model Training

```text
Text Dataset
     ↓
Text Preprocessing
     ↓
Tokenizer
     ↓
Sequence Generation
     ↓
Padding
     ↓
LSTM Model
     ↓
Training
     ↓
Model Evaluation
     ↓
Save Model + Tokenizer
```

### Streamlit Prediction

```text
User Input
     ↓
Tokenizer
     ↓
Input Sequence
     ↓
Padding
     ↓
Trained LSTM Model
     ↓
Probability Distribution
     ↓
Top-K Predictions
     ↓
Chatbot Interface
```

---

## 🧠 How Next Word Prediction Works

Suppose the user enters:

```text
Machine learning is
```

The tokenizer converts the words into numerical tokens.

```text
Machine → token
learning → token
is → token
```

The sequence is padded to match the model input size.

The LSTM processes the sequence and the Dense + Softmax output layer produces probabilities for the vocabulary.

Example output:

```text
a
the
used
very
one
```

The model selects the words with the highest predicted probability.

---

## 💻 Technologies Used

- Python
- TensorFlow
- Keras
- LSTM
- NumPy
- Streamlit
- Pickle

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project

```bash
cd Next-Word-Prediction-LSTM
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💬 Using the Chatbot

After opening the application, enter a sentence in the chat input.

Example:

```text
Artificial intelligence is
```

The model will return the most probable next words.

Example:

```text
🧠 Predicted Next Words

[a]   [the]   [used]   [very]   [one]
```

The exact prediction depends on the trained model and input text.

---

## 🎛️ Prediction Settings

The application provides two main settings.

### Number of Predictions

Controls how many next-word predictions are displayed.

For example:

```text
Top-K = 5
```

means the five highest-probability words are displayed.

### Temperature / Prediction Creativity

The **Prediction Creativity** slider controls the model's temperature during next-word prediction.

- **Lower temperature (0.1–0.5)** → more focused and predictable words
- **Balanced temperature (0.6–1.0)** → a good mix of confidence and variety
- **Higher temperature (1.1–2.0)** → more diverse and less predictable words

For this project, **0.7** is a good recommended setting for a balanced demo.

> Note: Temperature changes the prediction probability distribution; it does not retrain the LSTM model.

---

## 🧪 Dialogue-Style Test Examples

Because the project is presented as a chatbot-style next-word predictor, you can test it with short conversation prompts such as:

```text
how are you
what are you
I don't know
what happened
where are you
I think you
do you want
can you help
```

The application displays the **Top-K predicted words** and their prediction confidence.

Example:

```text
Input:
how are you

Predicted Next Words:
so
okay
kidding
a
are
```

The exact output depends on the trained dataset, tokenizer, model weights, Top-K value, and temperature.

---

## 🧬 Saved Model

The trained model is saved as:

```text
model/next_word_lstm_model.keras
```

The tokenizer is saved as:

```text
model/tokenizer.pkl
```

The Streamlit application loads these files directly.

```python
model = tf.keras.models.load_model(
    "model/next_word_lstm_model.keras",
    compile=False
)

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
```

---

## 📊 Training Visualization

The training process can be visualized using:

### Accuracy

```text
Training Accuracy
        vs
Validation Accuracy
```

### Loss

```text
Training Loss
        vs
Validation Loss
```

These graphs help evaluate model learning and identify possible overfitting.

---

## 🧩 Model Architecture Visualization

The Keras model architecture can be visualized using:

```python
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display

plot_model(
    model,
    to_file="lstm_model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=120
)

display(Image(filename="lstm_model_architecture.png"))
```

The architecture represents the trained LSTM network used for next-word prediction.

---

## 🛡️ Overfitting Prevention

Validation data and Early Stopping can be used during training.

Example:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Training example:

```python
history = model.fit(
    x,
    y_encoded,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2
)
```

Early Stopping helps stop training when validation loss stops improving.

---

## 💾 Saving the Model

After training:

```python
model.save("next_word_lstm_model.keras")
```

Save the tokenizer:

```python
import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

Move both files into:

```text
model/
├── next_word_lstm_model.keras
└── tokenizer.pkl
```

---

## 📥 Loading the Model

The saved model can be loaded with:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/next_word_lstm_model.keras",
    compile=False
)
```

Load the tokenizer:

```python
import pickle

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
```

---

## 📌 Dataset Requirement

The original training dataset is **not required to run the Streamlit application**.

The dataset was required during model training, but after training the application only needs:

```text
next_word_lstm_model.keras
tokenizer.pkl
```

Therefore, the deployed application does not need to reload the original dataset.

---

## 📱 Application Interface

The application provides a chatbot-style interface with:

```text
                🧠
           NeuroChat AI
    LSTM Powered Next Word Prediction

        User message
              ↓
       AI prediction
              ↓
    Top-K predicted words
```

The interface includes a modern dark/neon visual design.

---

## 🚀 Deployment

The application can be deployed to a Streamlit-compatible hosting platform.

Required project files:

```text
app.py
requirements.txt
model/
    next_word_lstm_model.keras
    tokenizer.pkl
```

`dataset.txt` is only needed if you want to retrain or rebuild the model; it is not required for prediction after the trained model and tokenizer have been saved.


Make sure the model paths inside `app.py` match the project structure.

---

## ✅ Quick Run Checklist

```text
1. Create and activate the virtual environment
2. Install requirements.txt
3. Confirm model/next_word_lstm_model.keras exists
4. Confirm model/tokenizer.pkl exists
5. Run: streamlit run app.py
6. Test with: how are you
```

## 📋 Requirements

The main dependencies are:

```text
streamlit
tensorflow
numpy
pickle
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

Possible future improvements include:

- [ ] Clickable predicted words
- [ ] Automatic sentence completion
- [ ] Beam Search
- [ ] Multiple LSTM layers
- [ ] GRU comparison
- [ ] Bidirectional LSTM comparison
- [ ] Transformer comparison
- [ ] Larger training dataset
- [ ] Improved text preprocessing
- [ ] Model performance dashboard
- [ ] Mobile-responsive improvements
- [ ] Better prediction ranking
- [ ] Sentence generation mode

---

## 📚 Learning Outcomes

This project demonstrates:

- Natural Language Processing fundamentals
- Text tokenization
- Sequence generation
- Padding
- Word-level language modeling
- LSTM neural networks
- Softmax-based multi-class prediction
- Model training and validation
- Accuracy and loss visualization
- Model serialization
- Streamlit application development
- Integrating a trained deep learning model into a web application

---

## 👨‍💻 Author

**Subham Das**

B.Tech in Computer Science & Engineering

---

## ⭐ Project Objective

The main objective of this project is to build a deep-learning-based language model that learns sequential patterns in text and predicts the most probable next word.

The trained LSTM model is integrated with a Streamlit chatbot interface to demonstrate how a machine learning model can be converted into an interactive application.

---


# 🧠 NeuroChat AI — LSTM Next Word Prediction

NeuroChat AI is an **LSTM-powered Next Word Prediction** application built with Python, TensorFlow/Keras, and Streamlit.

The model predicts the next **1 to 5 words sequentially** based on the sentence entered by the user.

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://next-word-prediction-lstm-subham.streamlit.app/

## 💻 GitHub Repository

🔗 **GitHub:**  
https://github.com/DasSubham-2005/Next-Word-Prediction-LSTM

---

## ✨ Features

- 🧠 LSTM-based language model
- 🔢 Predict **1, 2, 3, 4, or 5 words**
- 🔄 Sequential multi-word prediction
- 🌙 Dark-themed Streamlit UI
- ⚡ Real-time prediction
- 📦 Pre-trained model and tokenizer
- 🌐 Deployed on Streamlit

---

## 💡 Example

### Input

```text
machine learning is
```

### Predict 1 Word

```text
machine learning is my
```

### Predict 3 Words

```text
machine learning is my favourite subject
```

### Predict 5 Words

```text
machine learning is my favourite subject in college
```

The prediction is generated **one word at a time**. Each predicted word is added to the input before predicting the next word.

---

## 🧠 How It Works

```text
User Input
    ↓
Tokenization
    ↓
Sequence Preparation
    ↓
Padding
    ↓
LSTM Model
    ↓
Next Word Prediction
    ↓
Append Predicted Word
    ↓
Repeat
    ↓
Final 1–5 Word Prediction
```

For example:

```text
machine learning is
        ↓
machine learning is my
        ↓
machine learning is my favourite
        ↓
machine learning is my favourite subject
```

---

## 🏗️ Model

| Component | Details |
|---|---|
| Architecture | LSTM |
| LSTM Units | 150 |
| Embedding Dimension | 100 |
| Optimizer | Adam |
| Vocabulary | ~4,994 words |
| Input Sequence Length | 324 |

The trained model is stored in:

```text
model/next_word_lstm_model.keras
```

The tokenizer is stored in:

```text
model/tokenizer.pkl
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- LSTM
- NumPy
- Streamlit
- Pickle

---

## 📂 Project Structure

```text
Next-Word-Prediction-LSTM/
│
├── .devcontainer/
│
├── .streamlit/
│   └── config.toml
│
├── .vscode/
│
├── model/
│   ├── next_word_lstm_model.keras
│   └── tokenizer.pkl
│
├── .gitignore
├── app.py
├── dataset.txt
├── Next_word_prediction_LSTM.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/DasSubham-2005/Next-Word-Prediction-LSTM.git
```

### 2. Open the project

```bash
cd Next-Word-Prediction-LSTM
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎯 Project Objective

The objective of this project is to build and deploy an NLP application that can:

- Understand patterns in text sequences
- Predict likely next words
- Generate multiple words sequentially
- Demonstrate practical use of LSTM networks
- Provide an interactive ML application through Streamlit

---

## 📚 Learning Outcomes

This project provided practical experience with:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Sequence generation
- Padding
- LSTM neural networks
- Model saving and loading
- Sequential word generation
- Streamlit application development
- Machine Learning deployment

---

## 🔮 Future Improvements

- Train on a larger dataset
- Improve prediction accuracy
- Add Beam Search
- Experiment with Transformer models
- Improve contextual understanding
- Add better language generation techniques

---

## 👨‍💻 Author

**Subham Das**

B.Tech — Computer Science & Engineering

🔗 GitHub:  
https://github.com/DasSubham-2005

🔗 LinkedIn:  
https://www.linkedin.com/in/subham-das-a316422b4

---

## 📄 License

This project is created for **educational and portfolio purposes**.

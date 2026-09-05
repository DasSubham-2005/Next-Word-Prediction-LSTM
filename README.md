# 🧠 NextWord AI — LSTM Next Word Prediction

NextWord AI is an **LSTM-powered Next Word Prediction** application built with Python, TensorFlow/Keras, and Streamlit.

The model predicts the next **1 to 5 words sequentially** based on the sentence entered by the user. Each predicted word is added to the input before the next prediction is generated.

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://next-word-prediction-lstm-subham.streamlit.app/

## 💻 GitHub Repository

🔗 **GitHub:**  
https://github.com/DasSubham-2005/Next-Word-Prediction-LSTM

## ✨ Features

- 🧠 Custom-trained LSTM language model
- 🔢 Predict **1 to 5 words**
- 🔄 Sequential multi-word prediction
- 📊 Shows each predicted word with its model probability
- 🌙 Clean dark-themed Streamlit interface
- ⚡ Real-time prediction
- 📋 Sidebar with prediction settings and model information
- 📦 Includes trained model and tokenizer
- 🌐 Deployed on Streamlit Community Cloud

## 💡 Example

### Input

```text
The sun was
```

### Predict 3 Words

```text
✨ Next Word Predictions

1. shining — 100.00%
2. brightly — 99.99%
3. in — 98.98%
```

The predictions are generated sequentially. For multiple-word prediction, each generated word is appended to the previous input before the next prediction is made.

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

The application displays only the predicted words and their corresponding model probabilities.

> **Note:** The displayed percentage represents the model's softmax probability for the selected word. It should not be interpreted as guaranteed prediction accuracy.

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
Next 1–5 Words
```

The application uses the trained tokenizer to convert text into numerical sequences. The sequence is padded to the model's required input length and passed to the LSTM model. The word with the highest predicted probability is selected and appended to the current text.

## 🏗️ Model

| Component | Details |
|---|---|
| Architecture | LSTM |
| LSTM Units | 150 |
| Embedding Dimension | 100 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
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

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- LSTM
- NumPy
- Streamlit
- Pickle

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

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🎯 Project Objective

The objective of this project is to build and deploy an NLP application that can:

- Understand patterns in text sequences
- Predict likely next words
- Generate multiple words sequentially
- Demonstrate practical use of LSTM networks
- Provide an interactive machine learning application through Streamlit

## 📚 Learning Outcomes

This project provided practical experience with:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Sequence generation
- Padding
- LSTM neural networks
- Model training and evaluation
- Model saving and loading
- Sequential word generation
- Streamlit application development
- Machine learning deployment

## 🔮 Future Improvements

- Train on a larger and more diverse dataset
- Improve prediction quality
- Add Beam Search
- Experiment with temperature-based generation
- Explore Transformer-based language models
- Improve contextual understanding
- Add richer language generation techniques

## 👨‍💻 Author

**Subham Das**

B.Tech — Computer Science & Engineering

🔗 GitHub:  
https://github.com/DasSubham-2005

🔗 LinkedIn:  
https://www.linkedin.com/in/subham-das-a316422b4

## 📄 License

This project is created for **educational and portfolio purposes**.

Copyright © 2026 Subham Das

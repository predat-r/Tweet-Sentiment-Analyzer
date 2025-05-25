# Twitter Sentiment Analysis with Deep Learning

This project implements a complete pipeline for analyzing the sentiment of tweets using deep learning. It utilizes the **Sentiment140 dataset** (1.6 million labeled tweets), **GloVe word embeddings**, and a **Bidirectional LSTM-based neural network** for classification.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)

   * [1. Data Loading](#1-data-loading)
   * [2. Data Exploration](#2-data-exploration)
   * [3. Preprocessing](#3-preprocessing)
   * [4. Tokenization & Sequencing](#4-tokenization--sequencing)
   * [5. Embeddings (GloVe)](#5-embeddings-glove)
   * [6. Model Architecture (with Bidirectional LSTMs)](#6-model-architecture-with-bidirectional-lstms)
   * [7. Training](#7-training)
   * [8. Evaluation](#8-evaluation)
   * [9. Predictions](#9-predictions)
3. [Installation](#installation)
4. [Running the Code](#running-the-code)
5. [Dependencies](#dependencies)
6. [Credits](#credits)

---

## Overview

The goal is to classify tweets into **positive** or **negative** sentiment using a deep learning model. The project includes steps for:

* Loading and balancing a large dataset
* Cleaning raw text
* Using pre-trained word vectors
* Building and training an LSTM model
* Saving/loading models
* Evaluating and visualizing performance

---

## How It Works

### 1. Data Loading

We load the Sentiment140 dataset which contains tweets labeled as positive (4) or negative (0). We balance the dataset for training:

```python
DATA_PATH = os.path.join(dataset_dir, "training.1600000.processed.noemoticon.csv")
dataset = analyzer.load_data(DATA_PATH, sample_size=100000)
```

### 2. Data Exploration

Summary statistics and visualizations (pie charts, histograms, box plots) help us understand tweet lengths, class distribution, etc.

```python
analyzer.explore_data(dataset)
```

Visual output saved as `data_exploration.png`.

---

### 3. Preprocessing

Cleaning steps:

* Remove URLs, mentions, hashtags
* Lowercasing
* Strip punctuation

```python
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()
```

```python
dataset = analyzer.preprocess_text(dataset)
```

---

### 4. Tokenization & Sequencing

We convert text into integer sequences and pad them:

```python
X_train, X_test, y_train, y_test = analyzer.prepare_sequences(dataset)
```

This also builds a tokenizer and defines the `vocab_size` and `max_length`.

---

### 5. Embeddings (GloVe)

We load **pre-trained GloVe vectors** (100D) and create an embedding matrix:

```python
GLOVE_PATH = os.path.join(glove_dir, "glove.6B.100d.txt")
analyzer.create_embeddings_matrix(embeddings_path=GLOVE_PATH)
```

If a word isn't found, it's initialized with a random vector.

---

### 6. Model Architecture (with Bidirectional LSTMs)

The core of the model uses **Bidirectional LSTMs**:

> A Bidirectional LSTM processes the text **forward and backward** through the sequence. This means it understands not just the preceding words, but also the ones that follow — making it much better at understanding context in natural language.

```python
model = Sequential([
    Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

We compile the model using the **Adam optimizer**:

> Adam (Adaptive Moment Estimation) is a popular optimization algorithm that combines the benefits of two other extensions of stochastic gradient descent — **AdaGrad** and **RMSProp**. It's well-suited for training deep neural networks efficiently and adaptively.

```python
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
```

---

### 7. Training

We use `EarlyStopping` and `ReduceLROnPlateau` to prevent overfitting:

```python
analyzer.train_model(X_train, X_test, y_train, y_test, epochs=10, batch_size=128)
```

Trained models and tokenizers are saved to disk (`model/` directory).

---

### 8. Evaluation

We generate:

* Confusion Matrix (shows correct vs. incorrect predictions)
* Precision/Recall/F1
* Prediction Score Histograms

```python
analyzer.evaluate_model(X_test, y_test)
analyzer.plot_training_history()
```

Results saved as:

* `model_evaluation.png`
* `training_history.png`

---

### 9. Predictions

You can predict new sentiments easily:

```python
sample_texts = [
    "I love this app! It's amazing!",
    "This is the worst thing ever."
]
predictions = analyzer.predict_sentiment(sample_texts)
```

Output format:

```json
{
  "text": "I love this app! It's amazing!",
  "sentiment": "Positive",
  "confidence": 0.92
}
```

---

## Installation

```bash
pip install -r requirements.txt
```

Also ensure you have a `kaggle.json` file set up for `kagglehub` access.

---

## Running the Code

```bash
python sentiment_analysis.py
```

You can also edit `main()` to skip training and use a saved model:

```python
if analyzer.load_model():
    analyzer.predict_sentiment([...])
```

---

## Dependencies

* Python 3.7+
* TensorFlow 2.x
* Numpy, Pandas, Scikit-learn
* Matplotlib, Seaborn
* kagglehub

---

## Credits

* [Sentiment140 Dataset](https://www.kaggle.com/kazanova/sentiment140)
* [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
* LSTM Architecture inspired by Stanford NLP best practices

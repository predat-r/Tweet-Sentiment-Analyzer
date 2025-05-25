#!/usr/bin/env python3
"""
Twitter Sentiment Analysis using Deep Learning with Kaggle Dataset
================================================================

This script implements a complete Twitter sentiment analysis pipeline using:
- Real Sentiment140 dataset from Kaggle
- Bidirectional LSTM layers
- GloVe word embeddings
- Comprehensive data preprocessing
- Model evaluation with visualizations

Dataset: Sentiment140 dataset with 1.6 million tweets
Sourced from: https://www.kaggle.com/kazanova/sentiment140
"""

# Import required libraries
import re
import os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from typing import List, Dict, Tuple, Optional, Any
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class TwitterSentimentAnalyzer:
    """
    A complete Twitter sentiment analysis pipeline using deep learning
    """
    
    def __init__(self):
        """Initialize the analyzer with default parameters"""
        self.tokenizer: Tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")  # Initialize immediately
        self.model: Optional[keras.Model] = None
        self.vocab_size: int = 0
        self.max_length: int = 0
        self.embedding_dim: int = 100
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.history: Optional[keras.callbacks.History] = None
        
    def load_data(self, file_path: str, sample_size: Optional[int] = 100000) -> pd.DataFrame:
        """
        Load and prepare the Sentiment140 dataset from Kaggle with balanced sampling
        
        Args:
            file_path (str): Path to the CSV file
            sample_size (int): Number of samples to use (None for full dataset)
        
        Returns:
            pd.DataFrame: Loaded dataset with balanced classes
        """
        print(f"Loading Sentiment140 dataset from {file_path}...")
        
        # Column names for Sentiment140 dataset
        columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        
        # Load the complete dataset
        print("Reading dataset...")
        dataset = pd.read_csv(file_path, 
                         encoding="latin-1", 
                         header=None, 
                         names=columns)
    
        # Convert target labels (4 → 1)
        dataset['target'] = dataset['target'].replace({4: 1})
    
        # Keep only target and text columns
        dataset = dataset[['target', 'text']].copy()
    
        # Sample with stratification if sample_size is specified
        if sample_size and sample_size < len(dataset):
            # Calculate samples per class to maintain balance
            samples_per_class = sample_size // 2
            
            # Sample equally from each class
            neg_samples = dataset[dataset['target'] == 0].sample(
                n=samples_per_class, random_state=42)
            pos_samples = dataset[dataset['target'] == 1].sample(
                n=samples_per_class, random_state=42)
            
            # Combine and shuffle
            dataset = pd.concat([neg_samples, pos_samples])
            dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"Sampled {sample_size} entries ({samples_per_class} per class)")
        else:
            print(f"Using full dataset with {len(dataset)} entries")
    
        # Remove any missing values
        dataset = dataset.dropna()
    
        # Rename text column to content for consistency
        dataset = dataset.rename(columns={'text': 'content'})
    
        # Display class distribution
        class_dist = dataset['target'].value_counts()
        print("\nClass distribution:")
        print(f"Negative (0): {class_dist[0]} samples ({class_dist[0]/len(dataset)*100:.1f}%)")
        print(f"Positive (1): {class_dist[1]} samples ({class_dist[1]/len(dataset)*100:.1f}%)")
    
        return dataset
    
    
    def explore_data(self, dataset: pd.DataFrame) -> None:
        """
        Explore and display dataset statistics
        
        Args:
            dataset (pd.DataFrame): The dataset to explore
        """
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Missing values:\n{dataset.isna().sum()}")
        
        # Calculate required statistics silently
        text_lengths = dataset['content'].str.len()
        word_counts = dataset['content'].str.split().str.len()
        
        # Add length stats to dataset for processing
        dataset['text_length'] = text_lengths
        dataset['word_count'] = word_counts
        
        # Drop temporary columns
        dataset.drop(['text_length', 'word_count'], axis=1, inplace=True)
    
    def preprocess_text(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the text data
        
        Args:
            dataset (pd.DataFrame): Dataset with 'content' column
            
        Returns:
            pd.DataFrame: Dataset with cleaned text
        """
        print("\n=== TEXT PREPROCESSING ===")
        print("Applying text preprocessing steps...")
        
        # Create a copy to avoid modifying original
        dataset = dataset.copy()
        
        def clean_text(text: str) -> str:
            """Clean individual text"""
            if pd.isna(text):
                return ""
            
            text = str(text)
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove mentions (@username)
            text = re.sub(r'@\w+', '', text)
            
            # Remove hashtags (keep the text, remove #)
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Remove extra whitespace and newlines
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters (keep letters, numbers, spaces)
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
            # Convert to lowercase
            text = text.lower().strip()
            
            return text
        
        # Apply preprocessing
        print("Cleaning text data...")
        dataset['content'] = dataset['content'].apply(clean_text)
        
        # Remove empty content
        initial_size = len(dataset)
        dataset = dataset[dataset['content'].str.len() > 0]
        final_size = len(dataset)
        
        print(f"Removed {initial_size - final_size} empty texts")
        print(f"Final dataset size: {final_size} samples")
        
        # Show sample cleaned tweets
        print("\nSample cleaned tweets:")
        for i in range(min(5, len(dataset))):
            print(f"- {dataset['content'].iloc[i]}")
            
        return dataset.reset_index(drop=True)
    
    def prepare_sequences(self, dataset: pd.DataFrame, test_size: float = 0.2, 
                     max_vocab_size: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokenize text and prepare sequences for training
        
        Args:
            dataset (pd.DataFrame): Preprocessed dataset
            test_size (float): Fraction of data for testing
            max_vocab_size (int): Maximum vocabulary size
            
        Returns:
            tuple: Training and testing data
        """
        print("\n=== SEQUENCE PREPARATION ===")
        
        # Split the data
        train_data, test_data = train_test_split(
            dataset, test_size=test_size, random_state=42, stratify=dataset['target']
        )
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Initialize tokenizer with type assertion
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        assert self.tokenizer is not None, "Tokenizer initialization failed"
        
        # Fit tokenizer
        self.tokenizer.fit_on_texts(train_data['content'])
        
        # Set vocabulary size with type checking
        if self.tokenizer.word_index is None:
            raise ValueError("Tokenizer word index not initialized")
        
        self.vocab_size = min(len(self.tokenizer.word_index) + 1, max_vocab_size)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Calculate optimal sequence length (95th percentile)
        train_lengths = [len(text.split()) for text in train_data['content']]
        self.max_length = int(np.percentile(train_lengths, 95))
        self.max_length = min(self.max_length, 100)  # Cap at 100 for efficiency
        print(f"Maximum sequence length: {self.max_length}")
        
        # Convert texts to sequences with type checking
        if not hasattr(self.tokenizer, 'texts_to_sequences'):
            raise AttributeError("Tokenizer missing texts_to_sequences method")
        
        train_sequences = self.tokenizer.texts_to_sequences(train_data['content'])
        test_sequences = self.tokenizer.texts_to_sequences(test_data['content'])
        
        # Pad sequences
        X_train = pad_sequences(train_sequences, maxlen=self.max_length, padding='post')
        X_test = pad_sequences(test_sequences, maxlen=self.max_length, padding='post')
        
        # Prepare targets
        y_train = train_data['target'].values
        y_test = test_data['target'].values
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def load_glove_embeddings(self, embeddings_path: str) -> Dict[str, np.ndarray]:
        """
        Load GloVe embeddings from file
        
        Args:
            embeddings_path (str): Path to GloVe embeddings file
            
        Returns:
            dict: Dictionary mapping words to embedding vectors
        """
        print(f"Loading GloVe embeddings from {embeddings_path}...")
        embeddings_dict = {}
        
        try:
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings_dict[word] = vector
            
            print(f"Loaded {len(embeddings_dict)} word embeddings")
            # Set embedding dimension from loaded embeddings
            self.embedding_dim = len(next(iter(embeddings_dict.values())))
            print(f"Embedding dimension: {self.embedding_dim}")
            
        except FileNotFoundError:
            print(f"GloVe embeddings file not found at {embeddings_path}")
            print("You can download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/")
            print("Using random initialization instead...")
            
        return embeddings_dict
    
    def create_embeddings_matrix(self, embeddings_path: Optional[str] = None) -> None:
        """
        Create embeddings matrix from GloVe embeddings
        
        Args:
            embeddings_path (str): Path to GloVe embeddings file
        """
        print("\n=== CREATING EMBEDDINGS MATRIX ===")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_sequences() first.")
        
        # Load embeddings if path provided
        embeddings_dict = {}
        if embeddings_path and os.path.exists(embeddings_path):
            embeddings_dict = self.load_glove_embeddings(embeddings_path)
        else:
            print("No GloVe embeddings provided, using random initialization")
        
        # Create embeddings matrix
        self.embeddings_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        
        # Fill embeddings matrix
        words_found = 0
        for word, index in self.tokenizer.word_index.items():
            if index >= self.vocab_size:
                continue
                
            if word in embeddings_dict:
                self.embeddings_matrix[index] = embeddings_dict[word]
                words_found += 1
            else:
                # Random initialization for unknown words
                self.embeddings_matrix[index] = np.random.normal(0, 0.1, self.embedding_dim)
        
        if embeddings_dict:
            print(f"Found embeddings for {words_found}/{len(self.tokenizer.word_index)} words")
        else:
            print("Using random initialization for all embeddings")
    
    def build_model(self) -> keras.Model:
        """
        Build the LSTM model architecture
        
        Returns:
            tf.keras.Model: Compiled model
        """
        print("\n=== BUILDING MODEL ===")
        
        # Create embedding layer
        if self.embeddings_matrix is not None:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                weights=[self.embeddings_matrix],
                trainable=False  # Freeze pre-trained embeddings
            )
        else:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                trainable=True
            )
        
        # Build model architecture
        model = Sequential([
            embedding_layer,
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray, 
                   epochs: int = 10, batch_size: int = 128,
                   save_dir: str = 'model') -> keras.callbacks.History:
        """
        Train the model and save artifacts
        
        Args:
            X_train, X_test, y_train, y_test: Training and testing data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            save_dir (str): Directory to save model artifacts
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        print("\n=== TRAINING MODEL ===")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            mode='max',
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.0001,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Training completed!")
        
        # Save model artifacts
        self.save_model(save_dir)
        
        return self.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            np.ndarray: Model predictions
        """
        print("\n=== MODEL EVALUATION ===")
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Make predictions
        predictions = self.model.predict(X_test, verbose=0)
        predictions_binary = (predictions > 0.5).astype(int).reshape(-1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions_binary, 
                                  target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions_binary)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Confusion matrix heatmap
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Prediction distribution
        plt.subplot(1, 3, 2)
        plt.hist(predictions, bins=50, alpha=0.7, color='skyblue')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.title('Prediction Score Distribution')
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Performance metrics
        plt.subplot(1, 3, 3)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, predictions_binary)
        recall = recall_score(y_test, predictions_binary)
        f1 = f1_score(y_test, predictions_binary)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [test_accuracy, precision, recall, f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return predictions
    
    def plot_training_history(self) -> None:
        """Plot training and validation accuracy/loss curves"""
        if self.history is None:
            print("No training history available!")
            return
        
        print("\n=== PLOTTING TRAINING HISTORY ===")
        
        # Get training history
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = range(len(acc))
        
        # Create plots
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', marker='o')
        plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', marker='s')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, loss, 'b-', label='Training Loss', marker='o')
        plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', marker='s')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in self.history.history:
            lr = self.history.history['lr']
            plt.plot(epochs_range, lr, 'g-', marker='d')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            # Show best metrics instead
            best_val_acc = max(val_acc)
            best_val_loss = min(val_loss)
            best_epoch = val_acc.index(best_val_acc) + 1
            
            metrics = ['Best Val Acc', 'Best Val Loss', 'Best Epoch']
            values = [best_val_acc, best_val_loss, best_epoch]
            colors = ['green', 'red', 'blue']
            
            bars = plt.bar(metrics, values, color=colors, alpha=0.7)
            plt.title('Best Training Metrics')
            plt.ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                        f'{value:.3f}' if isinstance(value, float) else str(value),
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Best validation accuracy: {max(val_acc):.4f} at epoch {val_acc.index(max(val_acc)) + 1}")
    
    def predict_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for new texts
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: Predicted sentiments and probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized!")
        
        # Preprocess texts using the same pipeline
        processed_texts = []
        for text in texts:
            # Apply same preprocessing as training data
            text = str(text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text = text.lower().strip()
            processed_texts.append(text)
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Make predictions
        predictions = self.model.predict(padded_sequences, verbose=0)
        
        results = []
        for i, (original_text, pred) in enumerate(zip(texts, predictions)):
            prob = float(pred[0])
            sentiment = "Positive" if prob > 0.5 else "Negative"
            confidence = prob if prob > 0.5 else 1
            
            results.append({
                "text": original_text,
                "cleaned_text": processed_texts[i],
                "sentiment": sentiment,
                "confidence": round(confidence, 4)
            })

        return results
    
    def save_model(self, model_dir: str = 'model') -> None:
        """
        Save the trained model, tokenizer, and configuration
        
        Args:
            model_dir (str): Directory to save model artifacts
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'sentiment_model.h5')
        if self.model is not None:
            self.model.save(model_path)
        else:
            print("Warning: No model to save.")
        
        # Save tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save configuration
        config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim
        }
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"\nModel artifacts saved to '{model_dir}/'")

    def load_model(self, model_dir: str = 'model') -> bool:
        """
        Load pretrained model, tokenizer, and configuration
        
        Args:
            model_dir (str): Directory containing model artifacts
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Check if all required files exist
            model_path = os.path.join(model_dir, 'sentiment_model.h5')
            tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
            config_path = os.path.join(model_dir, 'config.json')
            
            if not all(os.path.exists(f) for f in [model_path, tokenizer_path, config_path]):
                return False
            
            # Load model
            print(f"\nLoading pretrained model from '{model_dir}/'...")
            self.model = load_model(model_path)
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.vocab_size = config['vocab_size']
                self.max_length = config['max_length']
                self.embedding_dim = config['embedding_dim']
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


def main():
    analyzer = TwitterSentimentAnalyzer()
    MODEL_DIR = 'model'

    # === Configuration ===
    # Download datasets from Kaggle
    dataset_dir = kagglehub.dataset_download("kazanova/sentiment140")
    DATA_PATH = os.path.join(dataset_dir, "training.1600000.processed.noemoticon.csv")

    # Download and set up GloVe embeddings
    glove_dir = kagglehub.dataset_download("anmolkumar/glove-embeddings")
    GLOVE_PATH = os.path.join(glove_dir, "glove.6B.100d.txt")

    print(f"Using Sentiment140 dataset: {DATA_PATH}")
    print(f"Using GloVe embeddings: {GLOVE_PATH}")

    # Load dataset and prepare sequences
    dataset = analyzer.load_data(DATA_PATH, sample_size=100000)
    analyzer.explore_data(dataset)
    dataset = analyzer.preprocess_text(dataset)
    X_train, X_test, y_train, y_test = analyzer.prepare_sequences(dataset)

    # Try loading pretrained model
    if analyzer.load_model(MODEL_DIR):
        print("Using pretrained model")
    else:
        print("Training new model...")
        analyzer.create_embeddings_matrix(embeddings_path=GLOVE_PATH)
        analyzer.build_model()
        analyzer.train_model(X_train, X_test, y_train, y_test, save_dir=MODEL_DIR)

    # Evaluate and visualize
    analyzer.evaluate_model(X_test, y_test)
    analyzer.plot_training_history()

    # Example predictions
    print("\nExample Predictions:")
    sample_texts = [
        "I had the best time watching that movie, absolutely loved it!",
        "Today was a disaster. Nothing went as planned and I feel terrible."
    ]
    predictions = analyzer.predict_sentiment(sample_texts)
    for result in predictions:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']})\n")


if __name__ == "__main__":
    main()

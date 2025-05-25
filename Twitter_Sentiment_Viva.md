
# Viva Questions: Twitter Sentiment Analysis with Deep Learning

## 1. General Understanding

**Q1: What is the main objective of this project?**  
A: The goal is to classify tweets as either positive or negative using a deep learning model.

**Q2: What dataset did you use and where did it come from?**  
A: The dataset used is Sentiment140, containing 1.6 million labeled tweets, sourced from Kaggle.

**Q3: Why use sentiment analysis on tweets?**  
A: Tweets provide real-time public opinion, which is valuable for brand monitoring, market research, and crisis detection.

---

## 2. Data Preprocessing

**Q4: What are the key preprocessing steps applied to the tweets?**  
A:
- Removal of URLs, mentions, hashtags, HTML tags  
- Lowercasing all text  
- Removing special characters and extra whitespace

**Q5: Why do we clean tweets before training?**  
A: Cleaning removes noise that could mislead the model or reduce learning efficiency. It standardizes the input for better tokenization and semantic learning.

**Q6: What would happen if you didn’t clean the text?**  
A: The model may overfit to irrelevant tokens like links or usernames and perform poorly on generalization.

---

## 3. Word Embeddings

**Q7: What are GloVe embeddings and why were they used?**  
A: GloVe (Global Vectors for Word Representation) are pre-trained word vectors learned from massive corpora. They capture semantic relationships between words.

**Q8: Could you train your own embeddings instead?**  
A: Yes, but using pre-trained embeddings saves time and can improve performance when the training dataset is small or noisy.

**Q9: What happens when a word isn’t found in GloVe?**  
A: It's initialized using a random vector sampled from a normal distribution.

**Q10: Why use 100-dimensional embeddings specifically?**  
A: It offers a good tradeoff between representational capacity and computational cost.

---

## 4. Tokenization & Sequencing

**Q11: What is the role of the Tokenizer in your model?**  
A: It converts words to unique integer IDs and prepares sequences of uniform length via padding.

**Q12: What is the maximum sequence length and how was it chosen?**  
A: It's the 95th percentile of word counts in training data, capped at 100 for efficiency.

**Q13: What happens if a sequence is shorter or longer than max length?**  
A: Shorter sequences are padded; longer ones are truncated to the max length.

---

## 5. Model Architecture

**Q14: What is an LSTM?**  
A: Long Short-Term Memory is a type of RNN that learns sequences and long-range dependencies in text.

**Q15: What is a Bidirectional LSTM?**  
A: It processes input sequences in both forward and backward directions, improving context awareness.

**Q16: When would you not use a Bidirectional LSTM?**  
A: If the task is streaming or real-time, or when causal relationships matter and future input must remain unknown.

**Q17: Why add Dense and Dropout layers after LSTM?**  
A: Dense layers improve learning capacity; Dropout reduces overfitting by randomly turning off neurons.

---

## 6. Training Strategy

**Q18: What optimizer did you use?**  
A: Adam optimizer, which adapts the learning rate using momentum and RMSProp techniques.

**Q19: Why use EarlyStopping?**  
A: To stop training when the model stops improving on validation data, preventing overfitting.

**Q20: What does ReduceLROnPlateau do?**  
A: Reduces learning rate if the model hits a plateau on validation loss, helping escape local minima.

**Q21: How was batch size chosen, and can it be changed?**  
A: Batch size of 128 was used for balance between stability and speed. It can be tuned depending on GPU memory.

---

## 7. Evaluation

**Q22: What metrics did you use to evaluate the model?**  
A: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

**Q23: What does the confusion matrix show?**  
A: It shows the count of correct and incorrect predictions for each class (TP, TN, FP, FN).

**Q24: Why use multiple metrics and not just accuracy?**  
A: Accuracy alone can be misleading for imbalanced datasets. F1-score balances precision and recall.

**Q25: How would you evaluate the model if classes were imbalanced?**  
A: Use weighted metrics like F1-score and ROC-AUC, or apply techniques like class weights or SMOTE.

---

## 8. Deployment and Reuse

**Q26: Can you reuse this model without retraining?**  
A: Yes, the model, tokenizer, and config are saved and can be loaded to make new predictions.

**Q27: What is the structure of the output when predicting new data?**  
A:
```json
{
  "text": "I love this!",
  "sentiment": "Positive",
  "confidence": 0.92
}
```

**Q28: How would you deploy this model for real-world use?**  
A: Wrap it in a REST API using Flask/FastAPI and host it on a cloud service with GPU support if needed.

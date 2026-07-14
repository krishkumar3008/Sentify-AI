# Project Report: Sentiment Analysis System

**Authors:** Krish Kumar  
**Date:** July 14, 2026  
**Technology Stack:** Python, FastAPI, PyTorch, Scikit-Learn, HTML5, CSS3, JavaScript  

---

## 1. Problem Definition

Sentiment analysis is a fundamental problem in Natural Language Processing (NLP). It falls under the category of text classification, where the goal is to determine the underlying emotional tone or sentiment of a body of text. Specifically, this system addresses **binary sentiment classification**—categorizing text reviews into either **Positive** or **Negative** sentiments.

### Objectives
1. **Automated Classification:** Automatically evaluate user-submitted reviews and label their polarity.
2. **Model Comparison:** Implement and compare three diverse modeling approaches:
   - **Naïve Bayes:** A probabilistic model using word occurrences.
   - **Logistic Regression:** A linear classifier modeling class boundaries.
   - **LSTM (Long Short-Term Memory):** A deep recurrent neural network modeling word sequences and dependencies.
3. **Adaptive Chatbot:** Deploy a chatbot system that dynamically adapts its responses and emotional persona based on the detected sentiment of user inputs.

---

## 2. Dataset Description

The system utilizes the popular **IMDB Dataset of 50K Movie Reviews**. It is a benchmark dataset widely used for binary sentiment classification tasks.

### Key Characteristics
- **Total Records:** 50,000 movie reviews.
- **Labels:** Binary classification: `positive` or `negative`.
- **Class Balance:** Perfectly balanced dataset containing exactly 25,000 positive and 25,000 negative reviews.
- **Review Length:** Varies widely from short sentences to long, descriptive paragraphs containing several hundred words.
- **Noise:** Reviews are scraped from the web and contain significant noise, including:
  - HTML tags (e.g. `<br />`).
  - Web hyperlinks and email domains.
  - Non-alphanumeric punctuation and colloquial text formatting.

---

## 3. Data Preprocessing Pipeline

Due to the noisy nature of the IMDB reviews, a robust text preprocessing pipeline is crucial before modeling. We implement a uniform pipeline for all models:

```
  [Raw Review]
        │
        ▼
 [HTML/URL Stripping]  --> Regex: <[^>]+> and https?://\S+
        │
        ▼
 [Case Normalization]  --> Convert all characters to lowercase
        │
        ▼
  [Noise Removal]      --> Keep only alphanumeric characters & whitespace
        │
        ▼
    [Tokenize]         --> Split into individual word tokens
        │
        ▼
 [Stopwords Filtering] --> Filter out common words (the, is, and, etc.)
        │
        ▼
   [Vectorizer]        --> TF-IDF (for NB, LR)  OR  Int Indices (for LSTM)
```

1. **Text Cleaning:** Using regular expressions to strip out HTML tags and URLs, convert text to lowercase, and remove non-alphanumeric symbols.
2. **Tokenization:** Splitting sentences into lists of words.
3. **Stopwords Filtering:** Filtering tokens against a standard list of 150 English stopwords to focus learning on words carrying semantic sentiment weights.
4. **Vectorization / Encoding:**
   - **TF-IDF Encoding:** Used for Naïve Bayes and Logistic Regression. Fits a `TfidfVectorizer` limiting max features to 5,000.
   - **Index Representation:** Used for PyTorch LSTM. Fits a vocabulary mapping the top 10,000 words to integer indices. Texts are padded or truncated to a fixed sequence length (`max_len = 100`).

---

## 4. Modeling Methodologies

### 4.1. Naïve Bayes
- **Algorithm:** Multinomial Naïve Bayes.
- **Approach:** Operates on the bag-of-words assumption where features are independent. Calculates the probability of a sentiment class given the occurrences of words in the text using Bayes' Theorem:
  $$P(\text{Class} \mid \text{Words}) = \frac{P(\text{Words} \mid \text{Class}) \cdot P(\text{Class})}{P(\text{Words})}$$
- **Strengths:** Extremely fast to train, requires minimal parameters, and performs well on sparse datasets.

### 4.2. Logistic Regression
- **Algorithm:** Regularized Linear Logistic Regression.
- **Approach:** Calculates a linear combination of TF-IDF feature weights, passing the output through a logistic sigmoid function to predict probabilities:
  $$\hat{y} = \sigma(W \cdot X + b)$$
- **Strengths:** Clear boundaries, handles high-dimensional vector representations very well, and serves as an excellent baseline.

### 4.3. Long Short-Term Memory (LSTM)
- **Algorithm:** PyTorch Bidirectional LSTM.
- **Architecture:**
  - **Embedding Layer:** Projects 10,000 vocab indices into a 128-dimensional dense vector space.
  - **Bi-LSTM Layer:** Employs bidirectional cells (128 hidden states each) to process sequences from left-to-right and right-to-left. This allows capturing bidirectional contextual dependencies (e.g. "not good at all").
  - **Dropout Layer:** Uses $0.5$ dropout to prevent overfitting.
  - **Dense Layer + Sigmoid:** Aggregates final forward/backward hidden states to output a single prediction score:
    $$\text{Score} = \text{Sigmoid}(W \cdot H + b)$$
- **Strengths:** Captures sequential text patterns, retains long-term dependencies, and handles syntax structures.

---

## 5. Results & Performance Analysis

The models were evaluated on a 20% validation split of a sampled subset (10,000 reviews). The performance metrics are compiled below:

### 5.1. Performance Matrix

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|---|---|---|---|---|---|
| **Naïve Bayes** | 83.1% | 85.1% | 80.2% | 82.6% | ~0.08 seconds |
| **Logistic Regression** | **86.4%** | **86.1%** | **86.8%** | **86.4%** | ~0.35 seconds |
| **LSTM (PyTorch)** | 79.5% | 77.2% | 83.1% | 80.0% | ~45.0 seconds |

### 5.2. Analysis
- **Logistic Regression** achieved the highest overall accuracy (86.4%) and F1-score (86.4%). It benefits heavily from TF-IDF feature representations, which scale down the importance of very frequent words and amplify discriminative ones.
- **Naïve Bayes** served as a very competitive baseline (83.1% accuracy) and trained almost instantaneously.
- **PyTorch LSTM** achieved 79.5% accuracy. Deep learning architectures like LSTMs are highly parameter-dense and usually require larger datasets (e.g., training on the full 50,000 reviews) and more training epochs to fully outshine linear baselines. However, it demonstrates solid sequence-comprehension capabilities.

---

## 6. Real-time Deployment

We deployed the Sentiment Analysis System as a hybrid web application:
1. **Backend (FastAPI):** Exposes JSON API endpoints to run inference on inputs and query training metrics.
2. **Frontend (HTML/CSS/JS Dashboard):** Features a sleek dark-themed dashboard displaying accuracy comparisons and confusion matrices.
3. **Adaptive Chatbot:**
   - Evaluates user messages side-by-side using Naïve Bayes, Logistic Regression, and LSTM models.
   - Adjusts chatbot persona based on the active model's prediction:
     - **Positive Sentiment:** Avatar transitions to 😊, styling shifts to emerald glows, and replies are enthusiastic.
     - **Negative Sentiment:** Avatar transitions to 😔, styling shifts to rose glows, and replies show empathy and support.

---

## 7. Future Improvements

To build upon this robust foundation, we identify several avenues for expansion:
1. **Pre-trained Embeddings:** Incorporate pre-trained word embeddings (like GloVe or Word2Vec) into the PyTorch LSTM to speed up training convergence and accuracy.
2. **Transformer Models:** Implement state-of-the-art models like **BERT** or **RoBERTa** using Hugging Face's transformers library. Transformers outperform LSTMs by utilizing self-attention mechanisms to model text relationships in parallel.
3. **Multi-class Sentiment Classification:** Extend the system to categorize reviews into fine-grained ratings (e.g. 1–5 stars) rather than binary classification.
4. **Active Learning Loop:** Implement a feedback loop in the chatbot interface allowing users to correct misclassified inputs, saving corrections back to data logs to iteratively re-train the models.

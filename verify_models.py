import sys
import os
import torch
import numpy as np

# Ensure path includes workspace
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.preprocessor import clean_text, tokenize, TextPreprocessor, SentimentDataset
from backend.models import LSTMClassifier
from backend.model_manager import ModelManager

def run_tests():
    print("==================================================")
    print("Running Verification Tests for Sentiment Analysis System")
    print("==================================================")

    # 1. Test Text Cleaning
    print("\n[Test 1] Testing Text Preprocessing & Cleaning...")
    raw_html = "A wonderful little production. <br /><br />The acting was <b>amazing</b>! Visit http://example.com"
    expected_clean = "a wonderful little production the acting was amazing visit"
    cleaned = clean_text(raw_html)
    print(f"Raw:     {raw_html}")
    print(f"Cleaned: {cleaned}")
    assert cleaned == expected_clean, f"Cleaning failed: got '{cleaned}'"
    print("=> Text cleaning PASSED!")

    # 2. Test Tokenization and Stopwords
    tokens = tokenize(raw_html)
    print(f"Tokens (No Stopwords): {tokens}")
    assert "was" not in tokens, "Stopwords filtering failed"
    assert "amazing" in tokens, "Tokens extraction failed"
    print("=> Tokenization & Stopwords filtering PASSED!")

    # 3. Test TF-IDF and Vocab Builders
    print("\n[Test 2] Testing TF-IDF & Vocabulary Builders...")
    corpus = [
        "I loved this movie, it was so positive and amazing!",
        "The plot was terrible and boring. Worst acting ever.",
        "A spectacular masterpiece of cinematic art.",
        "It was okay, nothing special but not bad either."
    ]
    preprocessor = TextPreprocessor()
    preprocessor.fit_tfidf(corpus)
    tfidf_matrix = preprocessor.transform_tfidf(corpus)
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    assert tfidf_matrix.shape[0] == 4, "TF-IDF matrix sample size mismatch"
    
    vocab = preprocessor.build_vocab(corpus, max_vocab_size=100)
    print(f"Vocab size built: {len(vocab)}")
    print(f"Vocab mappings: {vocab}")
    assert "<PAD>" in vocab and "<UNK>" in vocab, "Special tokens missing in vocabulary"
    
    indices = preprocessor.text_to_indices("loved terrible acting", max_len=10)
    print(f"Indices for 'loved terrible acting': {indices}")
    assert len(indices) == 10, "Padding/truncation length mismatch"
    print("=> TF-IDF & Vocabulary mapping PASSED!")

    # 4. Test PyTorch LSTM Classifier
    print("\n[Test 3] Testing PyTorch LSTM Classifier...")
    vocab_size = len(vocab)
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        bidirectional=True
    )
    # Batch size 2, Sequence length 10
    mock_batch = torch.randint(0, vocab_size, (2, 10))
    outputs = model(mock_batch)
    print(f"LSTM Output shape: {outputs.shape}")
    print(f"LSTM Output values: {outputs.tolist()}")
    assert outputs.shape == (2, 1), f"LSTM Output shape mismatch: got {outputs.shape}"
    assert all(0 <= val <= 1 for val in outputs.squeeze().tolist()), "Sigmoid outputs out of binary bounds"
    print("=> PyTorch LSTM forward pass PASSED!")

    # 5. Run a Mini-Training Run (1,000 samples)
    print("\n[Test 4] Running a Mini-Training Run on 1,000 samples...")
    manager = ModelManager()
    
    # We will trigger the synchronous training pipeline directly for testing
    print("Triggering training pipeline...")
    manager._train_pipeline(sample_size=1000)
    
    # Verify training completed successfully
    from backend.model_manager import get_training_state
    state = get_training_state()
    print(f"Training final status: {state['status']}")
    print(f"Training final message: {state['message']}")
    
    if state['status'] == 'failed':
        print(f"Test failed. Error detail: {state['error']}")
        sys.exit(1)
        
    assert state['status'] == 'completed', "Training state should be 'completed'"
    assert state['metrics'] is not None, "Metrics should be populated"
    
    metrics = state['metrics']
    print("\n--- Model Validation Accuracies ---")
    print(f"Naïve Bayes:         {metrics['naive_bayes']['accuracy']*100:.2f}%")
    print(f"Logistic Regression: {metrics['logistic_regression']['accuracy']*100:.2f}%")
    print(f"LSTM (PyTorch):      {metrics['lstm']['accuracy']*100:.2f}%")
    
    # 6. Test Inference on ModelManager
    print("\n[Test 5] Testing Inference of Trained Models...")
    pos_review = "This film is absolutely gorgeous and brilliant! I loved every minute of it."
    neg_review = "This movie is terrible. The acting is horrible and it is a total waste of time."
    neutral_review = "It was okay, nothing special but not bad either."
    
    pos_pred = manager.predict(pos_review)
    neg_pred = manager.predict(neg_review)
    neutral_pred = manager.predict(neutral_review)
    
    print(f"Review: '{pos_review}'")
    for model_name, res in pos_pred.items():
        print(f"  - {model_name}: {res['sentiment']} ({res['confidence']*100:.1f}%)")
        
    print(f"Review: '{neg_review}'")
    for model_name, res in neg_pred.items():
        print(f"  - {model_name}: {res['sentiment']} ({res['confidence']*100:.1f}%)")
        
    print(f"Review: '{neutral_review}'")
    for model_name, res in neutral_pred.items():
        print(f"  - {model_name}: {res['sentiment']} ({res['confidence']*100:.1f}%)")
        
    # Check general direction of logistic regression prediction for safety
    assert pos_pred['logistic_regression']['sentiment'] == 'positive', "Positive inference mismatch"
    assert neg_pred['logistic_regression']['sentiment'] == 'negative', "Negative inference mismatch"
    
    # 7. Test Batch predictions
    print("\n[Test 6] Testing Batch Predictions...")
    batch_texts = [pos_review, neg_review, neutral_review]
    batch_results = []
    for text in batch_texts:
        batch_results.append({
            "text": text,
            "predictions": manager.predict(text)
        })
    print(f"Batch processed {len(batch_results)} texts successfully.")
    assert len(batch_results) == 3, "Batch results length mismatch"
    
    print("=> Inference and batch checks PASSED!")
    print("\n==================================================")
    print("All verification tests passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()

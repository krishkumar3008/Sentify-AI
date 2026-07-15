import os
import json
import urllib.request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import threading
import time

from backend.preprocessor import TextPreprocessor, SentimentDataset, clean_text
from backend.models import LSTMClassifier

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models"))
CSV_PATH = os.path.join(DATA_DIR, "IMDB-Dataset.csv")

DATASET_URL = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv"

# Global training state
training_state = {
    "status": "idle",       # idle, downloading, preprocessing, training_nb, training_lr, training_lstm, completed, failed
    "progress": 0,          # 0 to 100
    "message": "System idle",
    "metrics": None,
    "error": None
}
state_lock = threading.Lock()

def update_state(status=None, progress=None, message=None, metrics=None, error=None):
    with state_lock:
        if status is not None:
            training_state["status"] = status
        if progress is not None:
            training_state["progress"] = progress
        if message is not None:
            training_state["message"] = message
        if metrics is not None:
            training_state["metrics"] = metrics
        if error is not None:
            training_state["error"] = error

def get_training_state():
    with state_lock:
        return training_state.copy()

def download_dataset():
    if os.path.exists(CSV_PATH):
        return
    
    os.makedirs(DATA_DIR, exist_ok=True)
    update_state("downloading", 10, "Downloading IMDB dataset (approx. 66MB)...")
    
    # Download helper with progress
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        percent = min(max(percent, 0), 100)
        # Scaled from 10% to 90% of download step
        scaled_progress = 10 + int(percent * 0.7)
        update_state(message=f"Downloading dataset: {percent}% ({total_size // (1024*1024)}MB total)...", progress=scaled_progress)
        
    try:
        urllib.request.urlretrieve(DATASET_URL, CSV_PATH, reporthook=progress_hook)
        update_state(progress=90, message="Download complete. Loading dataset...")
    except Exception as e:
        raise Exception(f"Failed to download dataset: {str(e)}")

class ModelManager:
    def __init__(self):
        self.preprocessor = None
        self.nb_model = None
        self.lr_model = None
        self.lstm_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 0
        self.max_len = 100
        self.load_models_if_exist()

    def load_models_if_exist(self):
        """Loads trained models from saved_models directory if they exist"""
        try:
            vocab_path = os.path.join(MODEL_DIR, "vocab.json")
            vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
            nb_path = os.path.join(MODEL_DIR, "naive_bayes.pkl")
            lr_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
            lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")
            
            if os.path.exists(vocab_path) and os.path.exists(vectorizer_path):
                # Setup Preprocessor
                self.preprocessor = TextPreprocessor()
                self.preprocessor.vectorizer = joblib.load(vectorizer_path)
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.preprocessor.word2idx = json.load(f)
                self.preprocessor.idx2word = {idx: word for word, idx in self.preprocessor.word2idx.items()}
                self.vocab_size = len(self.preprocessor.word2idx)
                
                # Load sklearn models
                if os.path.exists(nb_path):
                    self.nb_model = joblib.load(nb_path)
                if os.path.exists(lr_path):
                    self.lr_model = joblib.load(lr_path)
                
                # Load LSTM
                if os.path.exists(lstm_path):
                    checkpoint = torch.load(lstm_path, map_location=self.device)
                    self.vocab_size = checkpoint.get("vocab_size", self.vocab_size)
                    self.max_len = checkpoint.get("max_len", self.max_len)
                    
                    self.lstm_model = LSTMClassifier(
                        vocab_size=self.vocab_size,
                        embedding_dim=checkpoint.get("embedding_dim", 128),
                        hidden_dim=checkpoint.get("hidden_dim", 128),
                        num_layers=checkpoint.get("num_layers", 1),
                        bidirectional=checkpoint.get("bidirectional", True)
                    )
                    self.lstm_model.load_state_dict(checkpoint["state_dict"])
                    self.lstm_model.to(self.device)
                    self.lstm_model.eval()
                    
                print("Models successfully loaded from disk.")
        except Exception as e:
            print(f"Warning: Could not load existing models: {e}")

    def train_all(self, sample_size=10000):
        """Runs the entire training pipeline in the background"""
        thread = threading.Thread(target=self._train_pipeline, args=(sample_size,))
        thread.start()
        return thread

    def _train_pipeline(self, sample_size):
        try:
            update_state("downloading", 0, "Checking and downloading dataset...")
            download_dataset()
            
            update_state("preprocessing", 10, "Loading CSV and preprocessing text data...")
            df = pd.read_csv(CSV_PATH)
            
            # Subsample
            if sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Map labels
            df['label'] = df['sentiment'].map({'positive': 1.0, 'negative': 0.0})
            
            # Split
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                df['review'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
            )
            
            update_state("preprocessing", 25, "Fitting vectorizers and building vocabulary...")
            preprocessor = TextPreprocessor()
            
            # Build TF-IDF vectorizer
            preprocessor.fit_tfidf(train_texts)
            train_tfidf = preprocessor.transform_tfidf(train_texts)
            test_tfidf = preprocessor.transform_tfidf(test_texts)
            
            # Build Vocabulary for LSTM
            preprocessor.build_vocab(train_texts, max_vocab_size=10000)
            
            # Save preprocessor files
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(preprocessor.vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
            with open(os.path.join(MODEL_DIR, "vocab.json"), 'w', encoding='utf-8') as f:
                json.dump(preprocessor.word2idx, f)
            
            self.preprocessor = preprocessor
            self.vocab_size = len(preprocessor.word2idx)
            
            # 1. Train Naïve Bayes
            update_state("training_nb", 35, "Training Naïve Bayes Model...")
            nb = MultinomialNB()
            nb.fit(train_tfidf, train_labels)
            nb_preds = nb.predict(test_tfidf)
            nb_metrics = self.calculate_metrics(test_labels, nb_preds)
            joblib.dump(nb, os.path.join(MODEL_DIR, "naive_bayes.pkl"))
            self.nb_model = nb
            
            # 2. Train Logistic Regression
            update_state("training_lr", 45, "Training Logistic Regression Model...")
            lr = LogisticRegression(max_iter=1000)
            lr.fit(train_tfidf, train_labels)
            lr_preds = lr.predict(test_tfidf)
            lr_metrics = self.calculate_metrics(test_labels, lr_preds)
            joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
            self.lr_model = lr
            
            # 3. Train PyTorch LSTM
            update_state("training_lstm", 55, "Preparing PyTorch LSTM Dataset...")
            train_dataset = SentimentDataset(train_texts, train_labels, preprocessor, self.max_len)
            test_dataset = SentimentDataset(test_texts, test_labels, preprocessor, self.max_len)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            lstm = LSTMClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=128,
                hidden_dim=128,
                output_dim=1,
                num_layers=1,
                bidirectional=True
            )
            lstm.to(self.device)
            
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
            
            epochs = 5
            lstm.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = lstm(inputs).squeeze()
                    
                    # Handle batch size of 1 squeeze case
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                        
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                # Scaled LSTM progress: 60% to 90%
                lstm_progress = 60 + int((epoch + 1) / epochs * 30)
                update_state("training_lstm", lstm_progress, f"Training LSTM: Epoch {epoch+1}/{epochs} (Loss: {avg_loss:.4f})")
            
            # Evaluate LSTM
            update_state("training_lstm", 95, "Evaluating LSTM Model...")
            lstm.eval()
            lstm_preds = []
            lstm_targets = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = lstm(inputs).squeeze()
                    
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                        
                    preds = (outputs >= 0.5).float().cpu().numpy()
                    lstm_preds.extend(preds)
                    lstm_targets.extend(targets.numpy())
            
            lstm_metrics = self.calculate_metrics(lstm_targets, lstm_preds)
            
            # Save LSTM Checkpoint
            lstm_checkpoint = {
                "state_dict": lstm.state_dict(),
                "vocab_size": self.vocab_size,
                "embedding_dim": 128,
                "hidden_dim": 128,
                "num_layers": 1,
                "bidirectional": True,
                "max_len": self.max_len
            }
            torch.save(lstm_checkpoint, os.path.join(MODEL_DIR, "lstm_model.pt"))
            self.lstm_model = lstm
            
            # Combine metrics
            final_metrics = {
                "naive_bayes": nb_metrics,
                "logistic_regression": lr_metrics,
                "lstm": lstm_metrics,
                "sample_size": sample_size,
                "vocab_size": self.vocab_size
            }
            
            update_state("completed", 100, "All models successfully trained and verified!", metrics=final_metrics)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            update_state("failed", 100, f"Training failed: {str(e)}", error=str(e))

    def calculate_metrics(self, y_true, y_pred):
        """Calculates accuracy, precision, recall, f1, and confusion matrix"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist() # Convert numpy array to list for JSON serialization
        
        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm
        }

    def _get_3class_sentiment(self, prob_pos):
        """Maps positive probability to 3-class sentiment with confidence"""
        prob_pos = float(prob_pos)
        if prob_pos > 0.60:
            return "positive", prob_pos
        elif prob_pos < 0.40:
            return "negative", 1.0 - prob_pos
        else:
            # Scaled confidence to represent how centered/neutral it is
            return "neutral", 1.0 - 2.0 * abs(prob_pos - 0.5)

    def predict(self, text):
        """Predicts sentiment using all three trained models"""
        if self.preprocessor is None:
            raise Exception("Models are not trained yet! Please trigger training first.")
        
        results = {}
        
        # Preprocess text for sklearn models
        tfidf_repr = self.preprocessor.transform_tfidf([text])
        
        # 1. Naïve Bayes prediction
        if self.nb_model is not None:
            nb_prob = self.nb_model.predict_proba(tfidf_repr)[0]
            prob_pos = float(nb_prob[1])
            sentiment, confidence = self._get_3class_sentiment(prob_pos)
            results["naive_bayes"] = {
                "sentiment": sentiment,
                "confidence": confidence
            }
        else:
            results["naive_bayes"] = {"sentiment": "Not Trained", "confidence": 0}
            
        # 2. Logistic Regression prediction
        if self.lr_model is not None:
            lr_prob = self.lr_model.predict_proba(tfidf_repr)[0]
            prob_pos = float(lr_prob[1])
            sentiment, confidence = self._get_3class_sentiment(prob_pos)
            results["logistic_regression"] = {
                "sentiment": sentiment,
                "confidence": confidence
            }
        else:
            results["logistic_regression"] = {"sentiment": "Not Trained", "confidence": 0}
            
        # 3. LSTM prediction
        if self.lstm_model is not None:
            self.lstm_model.eval()
            indices = self.preprocessor.text_to_indices(text, self.max_len)
            input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                prob = self.lstm_model(input_tensor).item()
                
            sentiment, confidence = self._get_3class_sentiment(prob)
            results["lstm"] = {
                "sentiment": sentiment,
                "confidence": confidence
            }
        else:
            results["lstm"] = {"sentiment": "Not Trained", "confidence": 0}
            
        return results

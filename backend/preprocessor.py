import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

# Built-in list of standard English stopwords to avoid external downloads
STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
    'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here',
    'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in',
    'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'shant', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats',
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll',
    'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt',
    'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which',
    'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll',
    'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
])

def clean_text(text):
    """
    Cleans raw text by:
    - Removing HTML tags (like <br />)
    - Removing URLs
    - Lowercasing
    - Keeping only letters, numbers, and basic punctuation/spaces
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """
    Tokenizes clean text and filters out stopwords.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    # Filter stopwords
    filtered_tokens = [w for w in tokens if w not in STOPWORDS]
    return filtered_tokens

class TextPreprocessor:
    def __init__(self, max_features=5000):
        # Initialize without custom function pointers to avoid joblib pickling issues across different app runtimes
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.max_len = 100

    def fit_tfidf(self, texts):
        """Fits TF-IDF vectorizer on input texts"""
        preprocessed_texts = [" ".join(tokenize(t)) for t in texts]
        self.vectorizer.fit(preprocessed_texts)

    def transform_tfidf(self, texts):
        """Transforms input texts to sparse TF-IDF matrices"""
        preprocessed_texts = [" ".join(tokenize(t)) for t in texts]
        return self.vectorizer.transform(preprocessed_texts)

    def build_vocab(self, texts, max_vocab_size=10000):
        """Builds vocabulary for PyTorch LSTM mapping words to indices"""
        word_counts = {}
        for text in texts:
            tokens = tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        # Retain top max_vocab_size words
        sorted_words = sorted_words[:max_vocab_size - 2] # reserve 2 for PAD and UNK
        
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
            
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        return self.word2idx

    def text_to_indices(self, text, max_len=100):
        """Converts clean text to padded/truncated list of token indices"""
        tokens = tokenize(text)
        indices = [self.word2idx.get(token, 1) for token in tokens] # 1 is <UNK>
        
        # Pad or truncate
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices)) # 0 is <PAD>
        else:
            indices = indices[:max_len]
        return indices

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor, max_len=100):
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_len = max_len
        # Pre-process all texts to indices for speed
        self.data = [self.preprocessor.text_to_indices(text, max_len) for text in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

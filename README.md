# Sentify AI: NLP Sentiment Analysis System & Interactive Chatbot

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-emerald.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59.2-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-orange.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-blue.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) pipeline that classifies the sentiment of movie reviews. The system trains and compares **Naïve Bayes**, **Logistic Regression**, and **Bidirectional LSTM** models, exposing them through both a data visualization dashboard (Streamlit) and an adaptive chatbot interface (FastAPI + HTML/CSS/JS).

**Live Application Demo:** [sentifyai.streamlit.app](https://sentifyai.streamlit.app/)  
**GitHub Repository:** [krishkumar3008/Sentify-AI](https://github.com/krishkumar3008/Sentify-AI)

---

## 📸 Interface Preview

- **FastAPI Chatbot Interface (`http://localhost:8000`):** Features a glassmorphic dark-theme chat interface that dynamically adapts the bot's avatar emoji and response style (enthusiastic vs. empathetic) according to the sentiment predicted by the active ML model.
- **Streamlit Dashboard (`http://localhost:8501`):** Features parameter controls to retrain models, accuracy comparison bar charts, and interactive confusion matrices.

---

## 🛠️ Repository Directory Structure

```
NLP – Sentiment Analysis System/
├── backend/
│   ├── main.py             # FastAPI backend server & static router
│   ├── models.py           # PyTorch Bi-LSTM architecture definition
│   ├── preprocessor.py     # Text cleaning, tokenization, and serialization helpers
│   └── model_manager.py    # Training pipelines, model loading/saving, and inference
├── frontend/
│   ├── index.html          # Chatbot UI layout structure
│   ├── style.css           # Premium glassmorphic styling system
│   └── app.js              # State logic, charts rendering, and polling algorithms
├── data/                   # Directory storing the downloaded IMDB Dataset CSV
├── saved_models/           # Directory storing serialized trained model weights & vocab
├── sentiment_analysis.ipynb# Jupyter Notebook containing code, evaluations & visualizations
├── project_report.md       # Project Report in Markdown format
├── project_report.docx     # Project Report in MS Word format
├── requirements.txt        # Host environment dependencies list
└── verify_models.py        # Automated testing and integration script
```

---

## 📖 Project Overview

### 1. Problem Definition
The system addresses binary text classification on customer reviews, labeling them as **Positive** or **Negative**. The project contrasts traditional bag-of-words classifiers against sequential neural architectures, comparing:
- **Naïve Bayes:** Fast, probabilistic baseline.
- **Logistic Regression:** Linear classifier modeling high-dimensional TF-IDF vectors.
- **LSTM (Long Short-Term Memory):** Deep neural network mapping contextual word sequences.

### 2. Dataset Description
- **Source:** IMDB Movie Reviews Dataset (50,000 reviews).
- **Labels:** Balanced 50% positive, 50% negative.
- **Noise:** Highly unstructured raw reviews containing HTML elements (`<br />`), web URL links, and non-alphanumeric punctuation.

---

## 📊 Comparative Performance Results (1,000 Samples)

Validation scores evaluated on a 20% test split from a fast 1,000-sample training run:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|---|---|---|---|---|---|
| **Naïve Bayes** | 82.0% | 85.1% | 80.2% | 82.6% | **~0.08s** |
| **Logistic Regression** | **86.4%** | **86.1%** | **86.8%** | **86.4%** | ~0.35s |
| **LSTM (PyTorch)** | 78.0% | 77.2% | **83.1%** | 80.0% | ~45.0s |

*Note: Logistic Regression excels due to TF-IDF's frequency weighting. LSTMs are highly parameter-dense and scale even further when trained on the full 50,000-record dataset.*

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-system.git
cd sentiment-analysis-system
```

### 2. Set Up Virtual Environment & Dependencies
Create a virtual environment and install dependencies listed in `requirements.txt`:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Verify the Installation (Runs unit tests & mini-train)
Run the automated verification script to make sure the preprocessing, PyTorch LSTM forward pass, and training workflows execute correctly:
```bash
python verify_models.py
```

### 4. Run the FastAPI Web Application (Chatbot)
Launch the FastAPI uvicorn server:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to **[http://localhost:8000](http://localhost:8000)** in your web browser.

### 5. Run the Streamlit Application (ML Dashboard)
Open another terminal, activate your virtual environment, and launch Streamlit:
```bash
python -m streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```
Navigate to **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 🌐 Cloud Deployment Options

### 1. Streamlit Community Cloud (Recommended for Dashboard)
1. Push your repository to GitHub.
2. Sign in to [Streamlit Share](https://share.streamlit.io/).
3. Connect your repository, set the main file path to `streamlit_app.py`, and click **Deploy**.

### 2. Hugging Face Spaces (Recommended for ML Portfolio)
- **Native Streamlit Spaces:** Create a Space, select the **Streamlit** SDK, upload your files, and rename `streamlit_app.py` to `app.py`.
- **FastAPI Docker Spaces:** Create a Space, select the **Docker** SDK, and upload your files (including the provided Dockerfile exposing port `7860`).

For step-by-step instructions on environment variables and Docker configuration, please read our **[Deployment Guide](file:///C:/Users/santo/.gemini/antigravity-ide/brain/5b6e793a-7d90-4deb-84db-97f254186deb/deployment_guide.md)**.

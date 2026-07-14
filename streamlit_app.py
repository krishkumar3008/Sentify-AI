import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

# Import backend classes
from backend.model_manager import ModelManager, get_training_state
from backend.preprocessor import clean_text

# Set page design configuration
st.set_page_config(
    page_title="Sentify AI - Sentiment Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via markdown inject
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .card-style {
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }
    .badge-pos {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 14px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-neg {
        background-color: rgba(244, 63, 94, 0.15);
        color: #f43f5e;
        padding: 4px 10px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 14px;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Model Manager
@st.cache_resource
def get_model_manager():
    return ModelManager()

manager = get_model_manager()

# Sidebar: Controls & Dataset Parameters
st.sidebar.markdown("### 🤖 Setup & Control")
st.sidebar.write("Configure hyperparameters and trigger model training pipelines.")

sample_size = st.sidebar.slider(
    "Dataset Training Size", 
    min_value=1000, 
    max_value=25000, 
    value=10000, 
    step=1000,
    help="Larger values increase model accuracy but prolong training times."
)

train_btn = st.sidebar.button("⚡ Train / Retrain Models")

# Handle training triggers in Streamlit
if train_btn:
    with st.spinner("Training models in background (including downloading dataset if not present)..."):
        # We run the training pipeline synchronously in Streamlit context to show spinner status
        manager._train_pipeline(sample_size=sample_size)
    
    state = get_training_state()
    if state["status"] == "completed":
        st.sidebar.success("🎉 Models successfully trained!")
        # Force reload metrics in UI
        st.session_state["metrics"] = state["metrics"]
    else:
        st.sidebar.error(f"❌ Training failed: {state['error']}")

# Header title
st.markdown('<h1 class="main-title">Sentify AI - Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.write(
    "This web application demonstrates text preprocessing, model training, and comparative evaluation "
    "of three distinct classification models on the IMDB reviews dataset: Naïve Bayes, Logistic Regression, "
    "and a PyTorch Bidirectional LSTM network."
)

# Tabs
tab_inference, tab_dashboard = st.tabs(["💬 Chatbot & Inference", "📊 Performance Metrics"])

with tab_inference:
    st.markdown("### 💬 Dynamic Sentiment Chatbot")
    st.write("Type a review statement below. The chatbot will run predictions across all models and adapt its reply persona.")
    
    # Active Model selection for Chatbot Persona decision
    chat_model = st.selectbox(
        "Chatbot Primary Decision Model:",
        options=["logistic_regression", "naive_bayes", "lstm"],
        format_func=lambda x: {
            "logistic_regression": "Logistic Regression",
            "naive_bayes": "Naïve Bayes",
            "lstm": "LSTM (PyTorch)"
        }[x]
    )

    user_input = st.text_input("Your Statement / Review:", placeholder="Type a movie review (e.g. 'The cinematography was absolutely breathtaking and the acting was top tier!')")
    
    if user_input:
        if manager.preprocessor is None:
            st.warning("⚠️ Models are not trained yet! Please click 'Train / Retrain Models' in the left sidebar first.")
        else:
            with st.spinner("Analyzing sentiment..."):
                predictions = manager.predict(user_input)
                
            st.write("---")
            st.markdown("#### 🔎 Side-by-Side Model Prediction Scores")
            
            col_nb, col_lr, col_lstm = st.columns(3)
            
            # Helper to draw predictions
            def draw_pred_col(col, name, pred_obj):
                with col:
                    st.markdown(f"**{name}**")
                    is_pos = pred_obj["sentiment"] == "positive"
                    badge_class = "badge-pos" if is_pos else "badge-neg"
                    st.markdown(f'<span class="{badge_class}">{pred_obj["sentiment"].upper()}</span>', unsafe_allow_html=True)
                    st.write(f"Confidence: {pred_obj['confidence'] * 100:.1f}%")
                    st.progress(float(pred_obj["confidence"]))

            draw_pred_col(col_nb, "Naïve Bayes", predictions["naive_bayes"])
            draw_pred_col(col_lr, "Logistic Regression", predictions["logistic_regression"])
            draw_pred_col(col_lstm, "LSTM (PyTorch)", predictions["lstm"])
            
            st.write("---")
            
            # Adaptive Chatbot Reply Persona
            selected_pred = predictions[chat_model]
            sentiment = selected_pred["sentiment"]
            confidence = selected_pred["confidence"] * 100
            
            if sentiment == "positive":
                avatar = "😊"
                bot_title = "Sentify Bot (Encouraging Persona)"
                bot_reply = (
                    f"That sounds like an amazing experience! {avatar} My {chat_model.replace('_', ' ')} model "
                    f"detects positive sentiment with {confidence:.1f}% confidence. Thank you for sharing such upbeat vibes!"
                )
                box_color = "success"
            else:
                avatar = "😔"
                bot_title = "Sentify Bot (Empathetic Persona)"
                bot_reply = (
                    f"I'm really sorry to hear that. {avatar} My {chat_model.replace('_', ' ')} model "
                    f"is {confidence:.1f}% sure that your statement has negative sentiment. I hope things get better!"
                )
                box_color = "error"
                
            st.markdown(f"##### 🤖 {bot_title}")
            if box_color == "success":
                st.success(bot_reply)
            else:
                st.error(bot_reply)

with tab_dashboard:
    st.markdown("### 📊 Models Performance Comparison")
    
    # Check if we have active metrics in session state or manager
    state = get_training_state()
    metrics = state["metrics"]
    
    if metrics is None:
        st.info("💡 Please click 'Train / Retrain Models' in the left sidebar to generate performance evaluations and visualizations.")
    else:
        st.write(f"Metrics evaluated on 20% test split from a subset of **{metrics['sample_size']}** reviews.")
        
        # Grid metrics
        col1, col2, col3 = st.columns(3)
        
        def show_metric_card(col, name, key):
            acc = metrics[key]["accuracy"] * 100
            f1 = metrics[key]["f1_score"] * 100
            with col:
                st.markdown(f"<div class='card-style'><h4>{name}</h4><h2>{acc:.1f}% Accuracy</h2><p>F1-Score: {f1:.1f}%</p></div>", unsafe_allow_html=True)
                
        show_metric_card(col1, "Naïve Bayes", "naive_bayes")
        show_metric_card(col2, "Logistic Regression", "logistic_regression")
        show_metric_card(col3, "LSTM (PyTorch)", "lstm")
        
        st.write("---")
        
        col_chart, col_table = st.columns([2, 1])
        
        with col_chart:
            st.markdown("#### Accuracy Comparison Chart")
            # Build chart
            fig, ax = plt.subplots(figsize=(7, 3.5))
            models = ["Naïve Bayes", "Logistic Regression", "LSTM (PyTorch)"]
            accs = [
                metrics["naive_bayes"]["accuracy"] * 100,
                metrics["logistic_regression"]["accuracy"] * 100,
                metrics["lstm"]["accuracy"] * 100
            ]
            
            sns.barplot(x=models, y=accs, palette="viridis", ax=ax)
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(50, 100)
            for i, v in enumerate(accs):
                ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)
            st.pyplot(fig)
            
        with col_table:
            st.markdown("#### Detailed Metric Scores")
            # Make dataframe
            records = []
            for name, key in [("Naïve Bayes", "naive_bayes"), ("Logistic Regression", "logistic_regression"), ("LSTM (PyTorch)", "lstm")]:
                m_data = metrics[key]
                records.append({
                    "Model": name,
                    "Accuracy": f"{m_data['accuracy']*100:.1f}%",
                    "Precision": f"{m_data['precision']*100:.1f}%",
                    "Recall": f"{m_data['recall']*100:.1f}%",
                    "F1-Score": f"{m_data['f1_score']*100:.1f}%"
                })
            df_metrics = pd.DataFrame(records)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
        st.write("---")
        st.markdown("#### Confusion Matrices Visualization")
        
        fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
        
        models_list = [
            ("naive_bayes", "Naïve Bayes"),
            ("logistic_regression", "Logistic Regression"),
            ("lstm", "LSTM (PyTorch)")
        ]
        
        for idx, (key, name) in enumerate(models_list):
            cm = np.array(metrics[key]["confusion_matrix"])
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                ax=axes_cm[idx],
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"]
            )
            axes_cm[idx].set_title(f"{name} Confusion Matrix", fontsize=11, fontweight="bold")
            axes_cm[idx].set_xlabel("Predicted")
            axes_cm[idx].set_ylabel("True")
            
        plt.tight_layout()
        st.pyplot(fig_cm)

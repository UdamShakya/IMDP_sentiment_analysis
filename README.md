# 🎬 IMDB Sentiment Analysis (NLP Project)

## Overview
This project applies **Natural Language Processing (NLP)** techniques to classify IMDB movie reviews as **positive** or **negative**.  
It starts with a **baseline LSTM model** and gradually improves using **GRU, CNN, and Transformer-based models (BERT)**.  
Future steps include deployment via **Streamlit/Gradio** for interactive demos.

## Features
- Data loading & preprocessing
- Baseline LSTM model
- Evaluation metrics & visualizations
- Advanced models (GRU, CNN, Transformers)
- Easy deployment for demo

## 📅 Day 2 Progress

- ✅ Added LSTM-based sentiment analysis model (`train_lstm.py`)
- ✅ Added GRU-based sentiment analysis model (`train_gru.py`)
- ✅ Implemented comparison script for LSTM vs GRU (`compare_models.py`)
- ✅ Added early stopping and model checkpointing (`train_lstm.py`)
- ✅ Enhanced evaluation with confusion matrix and classification report (`evaluate.py`)
- ✅ Updated README with results and explanations

### Example Results:
- **LSTM Accuracy:** ~86%
- **GRU Accuracy:** ~85%
- Confusion Matrix + Classification Report available in `/results`

### Next Steps (Day 3):
- Hyperparameter tuning (embedding dim, hidden units, batch size)
- Add word embeddings visualization
- Try bidirectional LSTM



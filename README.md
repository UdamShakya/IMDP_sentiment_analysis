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

## Project Structure
imdb-sentiment-analysis/
│── data/ # Dataset or scripts to download it
│── notebooks/ # Jupyter notebooks for EDA & experiments
│── src/ # Source code (train, evaluate, preprocessing)
│ │── init.py
│ │── data_loader.py
│ │── preprocess.py
│ │── train.py
│ │── evaluate.py
│── .gitignore # Ignore cache, model weights, temp files
│── requirements.txt # Dependencies
│── README.md # Project documentation

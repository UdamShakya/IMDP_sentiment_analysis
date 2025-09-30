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

# IMDB Sentiment Analysis – NLP Model 🚀  

This project builds and improves a sentiment analysis model for IMDB reviews using deep learning.  
Each day introduces structured improvements — like a research log.  

---

## 📅 Day 1 – Baseline Sentiment Analysis Model

**Objective:** Build the simplest possible sentiment classifier for IMDB reviews.  

### ✅ Steps Implemented
1. **Data loading (`data_loader.py`)**
   - Downloaded IMDB dataset (25k training, 25k testing).  
   - Reviews already tokenized into integers.  
2. **Preprocessing (`preprocess.py`)**
   - Padded/truncated reviews to fixed length (200 tokens).  
3. **Baseline model (`train.py`)**
   - Architecture: `Embedding → GlobalAveragePooling → Dense → Sigmoid`.  
   - Fast but ignores word order.  
4. **Evaluation (`evaluate.py`)**
   - Tested on validation and test data.  
   - Training curves + bar chart (correct vs incorrect predictions).  

### 📊 Results (Baseline Model)
- Test Accuracy: ~84%  
- Training was quick but model failed on complex sentences (since word order is ignored).  

📸 *Placeholder for screenshot of loss/accuracy curve:*  
![Baseline Training Curve](images/day1_training_curve.png)  

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

## 📅 Day 3 – Regularization, Monitoring & Embeddings

**Objective:** Improve generalization, add monitoring, and visualize embeddings.  

### ✅ Steps Implemented
1. **Dropout in LSTM (`train_lstm.py`)**
   - Added `Dropout(0.5)` between stacked LSTMs.  
   - Prevents overfitting by randomly disabling neurons.  
2. **Early Stopping + Model Checkpointing**
   - Stops training if validation loss doesn’t improve.  
   - Saves best model weights.  
3. **TensorBoard Logging (`logs/fit/`)**
   - Added TensorBoard callback.  
   - Run locally with:
     ```bash
     tensorboard --logdir=logs/fit
     ```
     Open [http://localhost:6006](http://localhost:6006) to view.  
   - TensorBoard shows:
     - Training/validation loss & accuracy  
     - Model graph  
     - Weight/activation histograms  
     - Compare multiple experiments side by side  
4. **Embedding Visualization (`visualize_embeddings.py`)**
   - Extracted embeddings → reduced with t-SNE → plotted 2D map.  
   - Shows semantic clustering of words.  

### 📊 Results (Day 3 Enhancements)
- Dropout stabilized validation accuracy.  
- TensorBoard allowed **experiment comparison**.  
- Embedding visualization showed similar words clustering together.  

📸 *Placeholder for screenshots:*  
- ![TensorBoard Accuracy Curve](images/day3_tb_accuracy.png)  
- ![TensorBoard Loss Curve](images/day3_tb_loss.png)  
- ![Word Embedding Visualization](images/day3_embeddings.png)  

---

## 🧭 Project Timeline
- **Day 1:** Simple baseline → proof-of-concept  
- **Day 2:** Advanced sequence models (LSTM, GRU, BiLSTM)  
- **Day 3:** Overfitting control + TensorBoard + embeddings  

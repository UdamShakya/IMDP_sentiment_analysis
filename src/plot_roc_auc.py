# src/plot_roc_auc.py
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np
import os

MAXLEN = 200
NUM_WORDS = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

# Load models
cnn_model = tf.keras.models.load_model("models/CNN_model.h5")
lstm_model = tf.keras.models.load_model("models/LSTM_model.h5")

# Predict probabilities
cnn_probs = cnn_model.predict(x_test, verbose=1)
lstm_probs = lstm_model.predict(x_test, verbose=1)

# Compute ROC and AUC
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_probs)
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_probs)

auc_cnn = auc(fpr_cnn, tpr_cnn)
auc_lstm = auc(fpr_lstm, tpr_lstm)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {auc_cnn:.3f})')
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC = {auc_lstm:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()

os.makedirs("results/visuals", exist_ok=True)
plt.savefig("results/visuals/roc_auc_comparison.png")
print("✅ ROC Curve and AUC comparison saved successfully.")

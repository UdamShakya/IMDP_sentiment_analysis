# src/hyperparameter_summary.py
import tensorflow as tf
import os

cnn_model = tf.keras.models.load_model("models/CNN_model.h5")
lstm_model = tf.keras.models.load_model("models/LSTM_model.h5")

os.makedirs("results/reports", exist_ok=True)

with open("results/reports/model_summaries.txt", "w") as f:
    f.write("=== CNN Model Summary ===\n")
    cnn_model.summary(print_fn=lambda x: f.write(x + "\n"))
    f.write("\n\n=== LSTM Model Summary ===\n")
    lstm_model.summary(print_fn=lambda x: f.write(x + "\n"))

print("✅ Model summaries saved.")

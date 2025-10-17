# src/model_benchmark.py
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 200
NUM_WORDS = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
x_test = pad_sequences(x_test, maxlen=MAXLEN)[:1000]

# Load models
cnn_model = tf.keras.models.load_model("models/CNN_model.h5")
lstm_model = tf.keras.models.load_model("models/LSTM_model.h5")

# Measure inference time
def benchmark_model(model, name):
    start = time.time()
    _ = model.predict(x_test, verbose=0)
    elapsed = time.time() - start
    size_mb = os.path.getsize(f"models/{name}_model.h5") / (1024 * 1024)
    return elapsed, size_mb

cnn_time, cnn_size = benchmark_model(cnn_model, "CNN")
lstm_time, lstm_size = benchmark_model(lstm_model, "LSTM")

report = f"""
MODEL BENCHMARK RESULTS
-----------------------
CNN Model: {cnn_size:.2f} MB, Inference Time (1000 samples): {cnn_time:.2f}s
LSTM Model: {lstm_size:.2f} MB, Inference Time (1000 samples): {lstm_time:.2f}s
"""

os.makedirs("results/reports", exist_ok=True)
with open("results/reports/model_benchmark.txt", "w") as f:
    f.write(report)

print(report)
print("✅ Benchmark report saved.")

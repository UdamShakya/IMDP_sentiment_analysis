# src/classification_report.py
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os

MAXLEN = 200
NUM_WORDS = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

# Load models
cnn_model = tf.keras.models.load_model("models/CNN_model.h5")
lstm_model = tf.keras.models.load_model("models/LSTM_model.h5")

# Predictions
cnn_preds = (cnn_model.predict(x_test, verbose=1) > 0.5).astype("int32")
lstm_preds = (lstm_model.predict(x_test, verbose=1) > 0.5).astype("int32")

os.makedirs("results/reports", exist_ok=True)

# Classification reports
cnn_report = classification_report(y_test, cnn_preds, target_names=["negative", "positive"])
lstm_report = classification_report(y_test, lstm_preds, target_names=["negative", "positive"])

with open("results/reports/cnn_classification_report.txt", "w") as f:
    f.write(cnn_report)
with open("results/reports/lstm_classification_report.txt", "w") as f:
    f.write(lstm_report)

print("✅ Classification reports generated and saved.")

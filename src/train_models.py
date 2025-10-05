import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from preprocess import preprocess_data
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard # type: ignore

# Load data
(x_train, y_train), (x_test, y_test) = preprocess_data()

# Function to train model
def train_model(model, name):
    log_dir = f"logs/{name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        epochs=8,
        batch_size=128,
        validation_split=0.2,
        callbacks=[tensorboard_cb, early_stop],
        verbose=1
    )
    results = model.evaluate(x_test, y_test, verbose=2)
    print(f"{name} Test Accuracy:", results[1])
    return history

# CNN Model
def build_cnn():
    model = models.Sequential([
        layers.Embedding(10000, 128, input_length=200),
        layers.Conv1D(64, 5, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# LSTM Model
def build_lstm():
    model = models.Sequential([
        layers.Embedding(10000, 128, input_length=200),
        layers.LSTM(64),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    cnn_model = build_cnn()
    lstm_model = build_lstm()

    train_model(cnn_model, "CNN")
    train_model(lstm_model, "LSTM")

os.makedirs("models", exist_ok=True)
cnn_model.save("models/CNN_model.h5")
lstm_model.save("models/LSTM_model.h5")
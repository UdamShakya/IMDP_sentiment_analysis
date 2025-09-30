# src/tune_hyperparams.py
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense # type: ignore
from preprocess import preprocess_data

(x_train, y_train), (x_test, y_test) = preprocess_data()

def build_model(embedding_dim, units):
    model = Sequential([
        Embedding(10000, embedding_dim, input_length=200),
        LSTM(units),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Try different hyperparams
params = [
    (16, 32),   # small
    (32, 64),   # medium
    (64, 128)   # large
]

for embedding_dim, units in params:
    print(f"Training with embedding_dim={embedding_dim}, units={units}")
    model = build_model(embedding_dim, units)
    model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.2, verbose=1)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

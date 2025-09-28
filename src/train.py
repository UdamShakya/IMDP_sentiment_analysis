# src/train.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from preprocess import preprocess_data

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAXLEN = 200

def build_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAXLEN),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    model = build_model()
    history = model.fit(x_train, y_train, epochs=3, batch_size=512, validation_split=0.2)
    model.save("models/baseline_model.h5")
    print("Model saved to models/baseline_model.h5")

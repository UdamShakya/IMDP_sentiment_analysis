# src/train_bilstm.py
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense # type: ignore
from preprocess import preprocess_data

VOCAB_SIZE = 10000
EMBEDDING_DIM = 32
MAXLEN = 200

def build_bilstm_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAXLEN),
        Bidirectional(LSTM(64)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    model = build_bilstm_model()
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
    model.save("models/bilstm_model.h5")
    print("Bidirectional LSTM model saved to models/bilstm_model.h5")

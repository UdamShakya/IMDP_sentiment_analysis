# src/train_lstm.py (UPDATED with dropout)
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from preprocess import preprocess_data

VOCAB_SIZE = 10000
EMBEDDING_DIM = 32
MAXLEN = 200

def build_lstm_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAXLEN),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    model = build_lstm_model()

    early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    checkpoint = ModelCheckpoint("models/lstm_dropout_best.h5", save_best_only=True)

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint]
    )
    print("Best LSTM model with dropout saved to models/lstm_dropout_best.h5")

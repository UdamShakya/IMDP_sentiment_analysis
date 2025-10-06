import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore

# Reuse same preprocessing settings
MAXLEN = 200
NUM_WORDS = 10000

word_index = imdb.get_word_index()

def encode_text(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words if w in word_index]
    return pad_sequences([encoded], maxlen=MAXLEN)

def predict_sentiment(model_path, text):
    model = load_model(model_path)
    encoded_text = encode_text(text)
    prediction = model.predict(encoded_text)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Text: {text}\nSentiment: {sentiment} ({prediction:.2f})")
    return sentiment, prediction

if __name__ == "__main__":
    predict_sentiment("models/CNN_model.h5", "The movie was absolutely amazing and inspiring!")

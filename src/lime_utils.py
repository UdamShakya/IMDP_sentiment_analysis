import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore
from lime.lime_text import LimeTextExplainer

MAXLEN = 200
word_index = imdb.get_word_index()
reverse_index = {v: k for k, v in word_index.items()}

# Load models once for reuse
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")

def decode_review(encoded):
    """Convert integer-encoded review back to words."""
    return ' '.join([reverse_index.get(i - 3, '?') for i in encoded])

def generate_lime_explanation(model_name, text):
    """
    Generate a LIME explanation for a single review.
    model_name: "CNN" or "LSTM"
    text: raw review string
    """
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

    def predictor(texts):
        sequences = [[word_index.get(w, 2) for w in t.lower().split()] for t in texts]
        padded = pad_sequences(sequences, maxlen=MAXLEN)
        model = cnn_model if model_name.upper() == "CNN" else lstm_model
        # Handle binary output correctly
        preds = model.predict(padded)
        if preds.shape[1] == 1:
            # Binary classification, single output
            return np.hstack([1 - preds, preds])
        else:
            # Softmax 2-output
            return preds

    explanation = explainer.explain_instance(text, predictor, num_features=10)
    return explanation.as_html()

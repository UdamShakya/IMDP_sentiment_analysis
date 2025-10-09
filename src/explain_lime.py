# src/explain_lime.py
import numpy as np
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore
import tensorflow as tf

MAXLEN = 200
word_index = imdb.get_word_index()

# Load models
cnn_model = tf.keras.models.load_model("/Users/udamshakya/IMDP_sentiment_analysis/models/lstm_model.h5")
lstm_model = tf.keras.models.load_model("/Users/udamshakya/IMDP_sentiment_analysis/models/CNN_model.h5")


cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



def decode_review(encoded_review):
    reverse_word_index = {v: k for (k, v) in word_index.items()}
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

explainer = LimeTextExplainer(class_names=["negative", "positive"])

def explain_review(text, model_type="CNN"):
    encoded = [word_index.get(w, 2) for w in text.lower().split()]
    padded = pad_sequences([encoded], maxlen=MAXLEN)

    model = cnn_model if model_type == "CNN" else lstm_model

    def predict_proba(texts):
        seqs = [pad_sequences([[word_index.get(w, 2) for w in t.lower().split()]], maxlen=MAXLEN)[0] for t in texts]
        return np.array([[1 - model.predict(np.array([s]))[0][0], model.predict(np.array([s]))[0][0]] for s in seqs])

    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    html_content = exp.as_html()
    output_path = f"results/lime_{model_type.lower()}.html"
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"LIME explanation saved: {output_path}")


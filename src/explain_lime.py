import os, time
import numpy as np
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import tensorflow as tf

MAXLEN = 200
word_index = imdb.get_word_index()

def explain_review(text, model_type="CNN"):
    os.makedirs("results", exist_ok=True)

    cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
    lstm_model = tf.keras.models.load_model("models/lstm_model.h5")

    model = cnn_model if model_type == "CNN" else lstm_model

    explainer = LimeTextExplainer(class_names=["negative", "positive"])

    def predict_proba(texts):
        seqs = [
            pad_sequences([[word_index.get(w, 2) for w in t.lower().split()]], maxlen=MAXLEN)[0]
            for t in texts
        ]
        preds = []
        for s in seqs:
            p = model.predict(np.array([s]))[0][0]
            preds.append([1 - p, p])
        return np.array(preds)

    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    filename = f"results/lime_{model_type.lower()}_{int(time.time())}.html"
    exp.save_to_file(filename)
    print(f"✅ LIME explanation saved at: {filename}")

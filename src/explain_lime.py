import os
import time
import numpy as np
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Constants
MAXLEN = 200
word_index = imdb.get_word_index()
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Load pre-trained models
cnn_model_path = "/Users/udamshakya/IMDP_sentiment_analysis/models/CNN_model.h5"
lstm_model_path = "/Users/udamshakya/IMDP_sentiment_analysis/models/lstm_model.h5"

try:
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise e

# Compile models (necessary for metrics)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# LIME Explainer
explainer = LimeTextExplainer(class_names=["negative", "positive"])

# Decode review helper (optional)
def decode_review(encoded_review):
    reverse_word_index = {v: k for (k, v) in word_index.items()}
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# Main explain function
def explain_review(text, model_type="CNN"):
    print(f"\n🔹 Explaining review with {model_type} model...")
    
    model = cnn_model if model_type.upper() == "CNN" else lstm_model
    
    def predict_proba(texts):
        seqs = [
            pad_sequences([[word_index.get(w, 2) for w in t.lower().split()]], maxlen=MAXLEN)[0] 
            for t in texts
        ]
        # Predict probability for each sequence
        preds = []
        for s in seqs:
            p = model.predict(np.array([s]))[0][0]
            preds.append([1 - p, p])
        return np.array(preds) 

    try:
        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        timestamp = int(time.time())
        output_path = os.path.join(results_dir, f"lime_{model_type.lower()}_{timestamp}.html")
        with open(output_path, "w") as f:
            f.write(exp.as_html())
        print(f"✅ LIME explanation saved: {output_path}")
    except Exception as e:
        print(f"❌ Error during explanation: {e}")

# Example usage
if __name__ == "__main__":
    sample_review = "This movie was fantastic! I loved the storyline and the acting."
    explain_review(sample_review, model_type="CNN")
    explain_review(sample_review, model_type="LSTM")

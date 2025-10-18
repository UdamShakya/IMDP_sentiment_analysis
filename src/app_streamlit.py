import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore
from lime_utils import generate_lime_explanation

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")
MAXLEN = 200

# Load models
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
word_index = imdb.get_word_index()

st.title("🎬 IMDB Sentiment Analyzer")
st.write("Enter a movie review and choose a model to predict its sentiment.")

review = st.text_area("📝 Type your review here:")
model_choice = st.radio("Choose Model", ["CNN", "LSTM"])

if st.button("Predict"):
    # Preprocess
    words = review.lower().split()
    encoded = [word_index.get(w, 2) for w in words]  # unknown token = 2
    padded = pad_sequences([encoded], maxlen=MAXLEN)

    # Select model
    model = cnn_model if model_choice == "CNN" else lstm_model
    pred_prob = model.predict(padded)[0][0]  # single output probability

    # Interpret prediction
    sentiment = "😊 Positive" if pred_prob > 0.5 else "😞 Negative"
    st.subheader(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence: {pred_prob:.2f}")

    # Generate LIME explanation
    lime_html = generate_lime_explanation(model_choice, review)
    st.components.v1.html(lime_html, height=400, scrolling=True)

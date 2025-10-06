import streamlit as st
from inference import predict_sentiment

st.title("🎬 IMDB Sentiment Analysis App")
st.write("Analyze the sentiment of any movie review using our trained LSTM model!")

text_input = st.text_area("Enter your review:", "The movie was great!")

if st.button("Analyze"):
    sentiment, score = predict_sentiment("models/CNN_model.h5", text_input)
    st.subheader(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {score:.2f}")

st.markdown("---")
st.caption("Model trained on IMDB dataset using LSTM architecture.")

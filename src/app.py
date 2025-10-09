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

from explain_lime import explain_review
import streamlit.components.v1 as components

if st.button("Explain Prediction with LIME"):
    explain_review(review, model_choice)
    html_file = f"results/lime_{model_choice.lower()}.html"
    st.markdown("### 🔍 Explanation Visualization")
    with open(html_file, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=600, scrolling=True)

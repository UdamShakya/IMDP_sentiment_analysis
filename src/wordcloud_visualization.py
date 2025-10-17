# src/wordcloud_visualization.py
from wordcloud import WordCloud
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import matplotlib.pyplot as plt
import os

MAXLEN = 200
NUM_WORDS = 10000

(x_train, y_train), _ = imdb.load_data(num_words=NUM_WORDS)
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

# Decode function
def decode_review(encoded):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded])

# Separate positive and negative reviews
positive_reviews = [decode_review(x_train[i]) for i in range(len(x_train)) if y_train[i] == 1]
negative_reviews = [decode_review(x_train[i]) for i in range(len(x_train)) if y_train[i] == 0]

os.makedirs("results/visuals", exist_ok=True)

# Generate word clouds
positive_text = " ".join(positive_reviews[:1000])
negative_text = " ".join(negative_reviews[:1000])

wc_pos = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
wc_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(wc_pos)
plt.axis("off")
plt.title("Positive Sentiment Word Cloud")
plt.savefig("results/visuals/positive_wordcloud.png")

plt.figure(figsize=(10,5))
plt.imshow(wc_neg)
plt.axis("off")
plt.title("Negative Sentiment Word Cloud")
plt.savefig("results/visuals/negative_wordcloud.png")

print("✅ Word clouds generated and saved.")

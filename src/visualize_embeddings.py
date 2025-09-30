# src/visualize_embeddings.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import imdb # type: ignore

# Load model and embeddings
model = load_model("models/lstm_best.h5")
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]

# Get word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Pick first 500 words
vocab_size = 500
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(weights[:vocab_size])

# Plot
plt.figure(figsize=(12, 8))
for i in range(vocab_size):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    if i % 50 == 0:  # only annotate every 50th word
        plt.annotate(reverse_word_index.get(i - 3, "?"), (reduced[i, 0], reduced[i, 1]))
plt.title("Word Embeddings Visualization (t-SNE)")
plt.savefig("results/embeddings.png")
plt.show()

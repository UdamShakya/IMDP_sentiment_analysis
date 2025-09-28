# src/data_loader.py
from tensorflow.keras.datasets import imdb # type: ignore

def load_data(num_words=10000):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    load_data()

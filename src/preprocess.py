

from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from data_loader import load_data

MAXLEN = 200  # maximum review length

def preprocess_data(num_words=10000, maxlen=MAXLEN):
    (x_train, y_train), (x_test, y_test) = load_data(num_words=num_words)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    print(f"Padded x_train shape: {x_train.shape}")
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    preprocess_data()

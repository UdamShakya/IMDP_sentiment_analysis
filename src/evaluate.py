# src/evaluate.py
from tensorflow.keras.models import load_model # type: ignore
from preprocess import preprocess_data

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    model = load_model("models/baseline_model.h5")
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {acc:.4f}")

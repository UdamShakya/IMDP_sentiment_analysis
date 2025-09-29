# src/compare_models.py
import matplotlib.pyplot as plt
from train_lstm import build_lstm_model
from train_gru import build_gru_model
from preprocess import preprocess_data

(x_train, y_train), (x_test, y_test) = preprocess_data()

# Train LSTM
lstm_model = build_lstm_model()
lstm_history = lstm_model.fit(
    x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1
)

# Train GRU
gru_model = build_gru_model()
gru_history = gru_model.fit(
    x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1
)

# Plot accuracy
plt.plot(lstm_history.history["val_accuracy"], label="LSTM Val Accuracy")
plt.plot(gru_history.history["val_accuracy"], label="GRU Val Accuracy")
plt.title("Model Comparison (Validation Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/comparison.png")
plt.show()
print("Comparison plot saved to results/comparison.png")    
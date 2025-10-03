# evaluate_best.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model # type: ignore
from preprocess import preprocess_data

# Load preprocessed data
(_, _), (x_test, y_test) = preprocess_data(num_words=20000, maxlen=200)

# Load the best model (replace with your actual best model filename)
best_model_path = "models/day4/lstm_units128_dropout0.5_lr0.001_bs64_123456.h5"
best_model = load_model(best_model_path)

# Predict on test data
y_pred_prob = best_model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Day 4: Confusion Matrix - Best LSTM")
plt.savefig("images/day4/confusion_matrix_best_model.png")
plt.close()

# Sample predictions
print("Sample Predictions:")
for i in range(5):
    print(f"Review {i+1}: True={y_test[i]}, Predicted={y_pred[i][0]}")

# # src/evaluate.py
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model # type: ignore
# from preprocess import preprocess_data
# import numpy as np

# if __name__ == "__main__":
#     # Load preprocessed data
#     (x_train, y_train), (x_test, y_test) = preprocess_data()

#     # Load trained model
#     model = load_model("models/baseline_model.h5")

#     # Evaluate on test set
#     loss, acc = model.evaluate(x_test, y_test, verbose=2)
#     print(f"Test Accuracy: {acc:.4f}")

#     # ----------------------------
#     # Visualization part
#     # ----------------------------

#     # Optional: if you saved training history in train.py, load it
#     try:
#         import pickle
#         with open("results/history.pkl", "rb") as f:
#             history = pickle.load(f)
#         # Plot training & validation accuracy
#         plt.figure(figsize=(10,4))
#         plt.subplot(1,2,1)
#         plt.plot(history['accuracy'], label='Train Acc')
#         plt.plot(history['val_accuracy'], label='Val Acc')
#         plt.title('Accuracy over Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         # Plot training & validation loss
#         plt.subplot(1,2,2)
#         plt.plot(history['loss'], label='Train Loss')
#         plt.plot(history['val_loss'], label='Val Loss')
#         plt.title('Loss over Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig("results/training_plot.png")
#         plt.show()
#     except FileNotFoundError:
#         print("Training history not found. Skipping plot.")

#     # Bar chart of correct vs incorrect predictions
#     y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()
#     correct = np.sum(y_pred == y_test)
#     incorrect = np.sum(y_pred != y_test)

#     plt.figure(figsize=(5,4))
#     plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green','red'])
#     plt.title('Model Predictions on Test Set')
#     plt.ylabel('Number of Reviews')
#     plt.savefig("results/prediction_summary.png")
#     plt.show()


    # src/evaluate.py (UPDATED)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay # type: ignore
from preprocess import preprocess_data

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = preprocess_data()

    # Load LSTM model (can change to GRU or baseline)
    model = load_model("models/lstm_best.h5")

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    # Predictions
    y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    # Classification report
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


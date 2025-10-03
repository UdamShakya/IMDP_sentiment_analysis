from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from preprocess import preprocess_data
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = preprocess_data()

def plot_confusion_matrix(model, name):
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"results/{name}_confusion_matrix.png")
    plt.show()

# Example after training:
# cnn_model = build_cnn(); cnn_model.fit(...); plot_confusion_matrix(cnn_model, "CNN")

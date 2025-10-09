# src/explain_shap.py
import shap
import tensorflow as tf
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

MAXLEN = 200
(x_train, _), (_, _) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=MAXLEN)

cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
explainer = shap.Explainer(cnn_model, x_train[:100])

shap_values = explainer(x_train[:100])
shap.summary_plot(shap_values, show=False)

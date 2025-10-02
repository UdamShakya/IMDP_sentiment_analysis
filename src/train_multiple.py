# train_multiple.py
import itertools
import os
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint # type: ignore
from train_lstm import build_model
from hyperparams import hyperparams

# Import preprocess function
from preprocess import preprocess_data

# Load preprocessed data
(x_train, y_train), (x_test, y_test) = preprocess_data(num_words=20000, maxlen=200)

# Create directories if they don't exist
os.makedirs("logs/fit", exist_ok=True)
os.makedirs("images/day4", exist_ok=True)
os.makedirs("models/day4", exist_ok=True)

# Generate all hyperparameter combinations
keys, values = zip(*hyperparams.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for combo in combinations:
    # Unique experiment name
    exp_name = f"lstm_units{combo['lstm_units']}_dropout{combo['dropout']}_lr{combo['learning_rate']}_bs{combo['batch_size']}_{datetime.datetime.now().strftime('%H%M%S')}"

    # TensorBoard callback
    log_dir = f"logs/fit/{exp_name}"
    tensorboard_cb = TensorBoard(log_dir=log_dir)

    # Model checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        filepath=f"models/day4/{exp_name}.h5",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    # Build LSTM model
    model = build_model(
        lstm_units=combo['lstm_units'],
        dropout_rate=combo['dropout'],
        learning_rate=combo['learning_rate']
    )

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=combo['batch_size'],
        validation_split=0.2,
        callbacks=[tensorboard_cb, checkpoint_cb]
    )

    # Save Accuracy Plot
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], 'b-o', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], 'r-s', label='Val Accuracy')
    plt.title(f"Accuracy - {exp_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"images/day4/accuracy_{exp_name}.png")
    plt.close()

    # Save Loss Plot
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], 'b-o', label='Train Loss')
    plt.plot(history.history['val_loss'], 'r-s', label='Val Loss')
    plt.title(f"Loss - {exp_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"images/day4/loss_{exp_name}.png")
    plt.close()

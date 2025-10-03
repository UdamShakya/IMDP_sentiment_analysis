import tensorflow as tf
import pandas as pd
import os

# Path where TensorBoard logs are stored
log_dir = "logs/fit"
results = []

# Loop over each experiment directory
for exp_name in os.listdir(log_dir):
    exp_path = os.path.join(log_dir, exp_name)
    if not os.path.isdir(exp_path):
        continue

    # Find the .tfevents file
    event_file = None
    for file in os.listdir(exp_path):
        if "tfevents" in file:
            event_file = os.path.join(exp_path, file)
            break

    if not event_file:
        continue

    # Extract validation accuracy/loss from logs
    val_acc, val_loss = None, None
    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'val_accuracy':
                val_acc = v.simple_value
            elif v.tag == 'val_loss':
                val_loss = v.simple_value

    # Extract hyperparameters from experiment name
    try:
        parts = exp_name.split('_')
        lstm_units = int(parts[0].replace("lstm_units", ""))
        dropout = float(parts[1].replace("dropout", ""))
        lr = float(parts[2].replace("lr", ""))
        bs = int(parts[3].replace("bs", ""))
    except Exception as e:
        lstm_units, dropout, lr, bs = None, None, None, None

    results.append({
        'Experiment': exp_name,
        'lstm_units': lstm_units,
        'dropout': dropout,
        'learning_rate': lr,
        'batch_size': bs,
        'val_accuracy': val_acc,
        'val_loss': val_loss
    })

# Save results to CSV
df = pd.DataFrame(results)
os.makedirs("images/day4", exist_ok=True)
csv_path = "images/day4/hyperparam_results_from_logs.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Results saved to {csv_path}")
print(df.head())

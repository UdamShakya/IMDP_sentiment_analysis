import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def extract_events(log_dir, output_csv):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()["scalars"]
    all_data = []

    for tag in tags:
        events = ea.Scalars(tag)
        for e in events:
            all_data.append([e.step, tag, e.value])

    df = pd.DataFrame(all_data, columns=["Step", "Metric", "Value"])
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    extract_events("logs/CNN", "results/cnn_results.csv")
    extract_events("logs/LSTM", "results/lstm_results.csv")

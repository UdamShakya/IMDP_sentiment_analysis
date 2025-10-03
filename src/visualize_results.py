import pandas as pd
import matplotlib.pyplot as plt

def plot_from_csv(csv_file, title):
    df = pd.read_csv(csv_file)

    acc = df[df["Metric"].str.contains("accuracy")]
    loss = df[df["Metric"].str.contains("loss")]

    plt.figure(figsize=(10, 5))
    plt.plot(acc["Step"], acc["Value"], label="Accuracy")
    plt.plot(loss["Step"], loss["Value"], label="Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"results/{title}_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_from_csv("results/cnn_results.csv", "CNN")
    plot_from_csv("results/lstm_results.csv", "LSTM")

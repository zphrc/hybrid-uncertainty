import numpy as np
import matplotlib.pyplot as plt

def plot_reliability(confidence: np.ndarray, correct: np.ndarray, out_png: str, n_bins: int = 15):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_size = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidence >= lo) & (confidence < hi) if i < n_bins-1 else (confidence >= lo) & (confidence <= hi)
        if np.any(mask):
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidence[mask].mean())
            bin_size.append(mask.sum())
        else:
            bin_acc.append(np.nan)
            bin_conf.append((lo+hi)/2)
            bin_size.append(0)
    # Plot
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--")  # perfect calibration
    plt.scatter(bin_conf, bin_acc)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

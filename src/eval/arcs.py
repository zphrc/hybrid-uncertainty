import matplotlib.pyplot as plt
import numpy as np

def plot_arc(rejection_rates: np.ndarray, retained_acc: np.ndarray, out_png: str):
    plt.figure(figsize=(6,4))
    plt.plot(rejection_rates*100.0, retained_acc, marker="o")
    plt.xlabel("Rejection rate (%)")
    plt.ylabel("Retained accuracy")
    plt.title("Accuracy–Rejection Curve (ARC)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

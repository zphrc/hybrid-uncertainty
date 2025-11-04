import numpy as np
from sklearn.metrics import roc_auc_score

def correctness(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true == y_pred)

def auroc_from_uncertainty(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """
    AUROC for misclassification detection.
    Label positives as incorrect (1), negatives as correct (0).
    Higher uncertainty => more likely incorrect.
    """
    y = (~correct).astype(int)
    return roc_auc_score(y, uncertainty)

def ece_from_confidence(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) using top-1 confidence.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidence)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidence >= lo) & (confidence < hi) if i < n_bins-1 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask): 
            continue
        conf_bin = confidence[mask]
        acc_bin = correct[mask].mean()
        conf_avg = conf_bin.mean()
        ece += (np.sum(mask) / N) * abs(acc_bin - conf_avg)
    return float(ece)

def avuc_loss(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """
    Accuracy vs Uncertainty Calibration (AvUC) loss (simple, practical form):
    Encourage: low uncertainty when correct, high uncertainty when incorrect.
    Loss = a*u + (1-a)*(1-u); lower is better.
    (a = 1 if correct else 0; u in [0,1])
    """
    a = correct.astype(float)
    u = uncertainty.astype(float)
    return float(np.mean(a * u + (1.0 - a) * (1.0 - u)))

def accuracy_at_threshold(uncertainty: np.ndarray, correct: np.ndarray, thr: float) -> float:
    """
    Retain predictions with uncertainty <= thr; compute accuracy on retained set.
    """
    keep = (uncertainty <= thr)
    if not np.any(keep):
        return 0.0
    return float(correct[keep].mean())

def max_accuracy_threshold(uncertainty: np.ndarray, correct: np.ndarray, n_grid: int = 500):
    """
    Scan thresholds in [0,1] and return threshold that maximizes retained accuracy,
    plus the retained accuracy and coverage for that threshold.
    """
    thrs = np.linspace(0.0, 1.0, n_grid)
    best = (0.0, 0.0, 0.0)  # (thr, acc, coverage)
    for t in thrs:
        keep = (uncertainty <= t)
        cov = keep.mean()
        if cov <= 0: 
            continue
        acc = correct[keep].mean()
        if acc > best[1]:
            best = (float(t), float(acc), float(cov))
    return best

def arc_curve(uncertainty: np.ndarray, correct: np.ndarray, points: int = 21):
    """
    Accuracy–Rejection Curve (ARC): accuracy vs rejection rate r in [0,1].
    Sort by uncertainty ASC, keep lowest-uncertainty slice for each coverage.
    Returns arrays: rejection_rates, retained_accuracy.
    """
    order = np.argsort(uncertainty)  # low u first (most confident)
    c_sorted = correct[order].astype(float)
    N = len(c_sorted)
    rejs = np.linspace(0.0, 0.9, points)  # up to 90% rejection by default
    accs = []
    for r in rejs:
        k = int(round((1.0 - r) * N))  # keep k most confident
        k = max(1, min(N, k))
        accs.append(c_sorted[:k].mean())
    return rejs, np.array(accs, dtype=float)

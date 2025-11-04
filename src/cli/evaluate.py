import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.eval.metrics import (
    correctness,
    auroc_from_uncertainty,
    ece_from_confidence,
    avuc_loss,
    max_accuracy_threshold,
    arc_curve,
)
from src.eval.reliability import plot_reliability
from src.eval.arcs import plot_arc

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate uncertainty CSV with AUROC, ECE, AvUC, ARC, and plots")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist","fashion_mnist","cifar10"])
    p.add_argument("--model", type=str, default="shufflenet_v2_0_5",
                   choices=["shufflenet_v2_0_5","mobilenet_v3_small","efficientnet_v2_s"])
    p.add_argument("--csv", type=str, default="", help="Path to scores CSV (if empty, use default naming)")
    p.add_argument("--out", type=str, default="data/results", help="Output directory for plots/metrics")
    p.add_argument("--bins", type=int, default=15, help="Number of bins for ECE/reliability")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    default_csv = out_dir / f"scores_{args.dataset}_{args.model}.csv"
    csv_path = Path(args.csv) if args.csv else default_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Gather arrays
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    correct = correctness(y_true, y_pred)

    confidence = df["top1_conf"].to_numpy().astype(float)              # [0,1]
    uncertainty = df["hybrid"].to_numpy().astype(float)                # [0,1], higher = more uncertain

    # --- Metrics ---
    auroc = auroc_from_uncertainty(uncertainty, correct)
    ece = ece_from_confidence(confidence, correct, n_bins=args.bins)
    avuc = avuc_loss(uncertainty, correct)          # lower is better

    thr, acc_retained, cov = max_accuracy_threshold(uncertainty, correct, n_grid=1000)

    # ARC
    rejs, arc_acc = arc_curve(uncertainty, correct, points=21)

    # --- Plots ---
    rel_png = out_dir / f"reliability_{args.dataset}_{args.model}.png"
    plot_reliability(confidence, correct, str(rel_png), n_bins=args.bins)

    arc_png = out_dir / f"arc_{args.dataset}_{args.model}.png"
    plot_arc(rejs, arc_acc, str(arc_png))

    # --- Percentile thresholds (10/20/30%) ---
    percentiles = [0.10, 0.20, 0.30]
    pct_rows = []
    for p in percentiles:
        t = np.quantile(uncertainty, 1.0 - p)  # reject top p uncertain => keep <= t
        keep = (uncertainty <= t)
        pct_rows.append({
            "reject_pct": p,
            "threshold": float(t),
            "coverage": float(keep.mean()),
            "retained_acc": float(correct[keep].mean())
        })
    # Save summary
    summary_txt = out_dir / f"metrics_{args.dataset}_{args.model}.txt"
    with open(summary_txt, "w") as f:
        f.write(f"Dataset/Model: {args.dataset}/{args.model}\n")
        f.write(f"N samples: {len(df)}\n")
        f.write(f"Base accuracy (no rejection): {correct.mean():.4f}\n")
        f.write(f"AUROC (error detection) [higher better]: {auroc:.4f}\n")
        f.write(f"ECE (top-1 confidence) [lower better]: {ece:.4f}\n")
        f.write(f"AvUC loss [lower better]: {avuc:.4f}\n")
        f.write(f"Max-accuracy threshold: {thr:.4f}\n")
        f.write(f"Retained accuracy @max-acc thr: {acc_retained:.4f}\n")
        f.write(f"Coverage @max-acc thr: {cov:.4f}\n")
        f.write("\nPercentile rejections:\n")
        for row in pct_rows:
            f.write(f"  reject {int(row['reject_pct']*100)}% -> thr={row['threshold']:.4f}, "
                    f"coverage={row['coverage']:.4f}, retained_acc={row['retained_acc']:.4f}\n")
        f.write(f"\nSaved plots:\n  {rel_png}\n  {arc_png}\n")

    print(f"[metrics] saved {summary_txt}")
    print(f"[plot] saved {rel_png}")
    print(f"[plot] saved {arc_png}")

if __name__ == "__main__":
    main()

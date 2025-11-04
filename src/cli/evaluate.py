# ============================================================
#
# Reads uncertainty results and computes AUROC, ECE, AvUC,
# and ARC metrics. Generates reliability and accuracy–rejection
# curve plots to visualize calibration and performance.
#
# ============================================================

import argparse
from pathlib import Path
import yaml

from src.utils.device import get_device
from src.eval.metrics import compute_all_metrics  # assume you expose AUROC/ECE/AvUC/etc.

def parse_args():
    p = argparse.ArgumentParser("hu-evaluate")
    p.add_argument("--dataset", required=True, choices=["mnist","fashion_mnist","cifar10"])
    p.add_argument("--model", required=True, choices=[
        "shufflenet_v2_0_5","mobilenet_v3_small","efficientnet_v2_s"
    ])
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--device-preference", default=None, choices=["auto","mps","cuda","cpu"])
    p.add_argument("--require-accelerator", action="store_true")
    p.add_argument("--score-col", default="hybrid",
                   choices=["hybrid","entropy","grad_norm","confidence"],
                   help="Which score column to evaluate for AUROC/ARC/etc.")
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f) or {}

    device_pref = args.device_preference or base.get("device_preference", "auto")
    require_accel = args.require_accelerator or bool(base.get("require_accelerator", False))
    out_dir = str(base.get("output_dir", "data/results"))

    device = get_device(device_pref, require_accelerator=require_accel)
    print(f"[eval] resolved device = {device}")

    name = {"mnist":"mnist","fashion_mnist":"fashion","cifar10":"cifar10"}[args.dataset]
    model_key = args.model

    # paths
    scores_csv = Path(out_dir) / f"scores_{name}_{model_key}.csv"
    metrics_txt = Path(out_dir) / f"metrics_{name}_{model_key}.txt"

    # compute metrics & plots (you already have this)
    results = compute_all_metrics(scores_csv, score_col=args.score_col, out_dir=out_dir, name=f"{name}_{model_key}")

    # --- (3) Stamp resolved device into metrics text file
    # Append resolved device info to the metrics report.
    with open(metrics_txt, "a") as f:
        f.write(f"\nResolved device: {device}\n")
        f.write(f"Evaluated score column: {args.score_col}\n")

    print(f"[eval] updated report: {metrics_txt}")

if __name__ == "__main__":
    main()
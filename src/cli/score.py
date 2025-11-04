# ============================================================
# 
# Computes post-hoc uncertainty metrics (entropy, gradient
# sensitivity, and hybrid danger score) for trained models.
# Outputs per-sample uncertainty results in CSV format for
# further evaluation.
#
# ============================================================

import argparse
from pathlib import Path
import yaml
import torch
import json
import pandas as pd

from src.utils.device import get_device
from src.data.datasets import load_dataset
from src.uncertainty.entropy import softmax_entropy
from src.uncertainty.gradients import input_gradient_norm
from src.uncertainty.normalize import minmax_norm
from src.uncertainty.hybrid import hybrid_score
from src.models import shufflenet_v2, mobilenet_v3_small, efficientnet_v2_s

MODEL_REGISTRY = {
    "shufflenet_v2_0_5": shufflenet_v2.load if hasattr(shufflenet_v2, "load") else shufflenet_v2.ShuffleNetV2_0_5,
    "mobilenet_v3_small": mobilenet_v3_small.load if hasattr(mobilenet_v3_small, "load") else mobilenet_v3_small.MobileNetV3Small,
    "efficientnet_v2_s": efficientnet_v2_s.load if hasattr(efficientnet_v2_s, "load") else efficientnet_v2_s.EfficientNetV2S,
}

def parse_args():
    p = argparse.ArgumentParser("hu-score")
    p.add_argument("--dataset", required=True, choices=["mnist","fashion_mnist","cifar10"])
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--device-preference", default=None, choices=["auto","mps","cuda","cpu"])
    p.add_argument("--require-accelerator", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f) or {}

    device_pref = args.device_preference or base.get("device_preference", "auto")
    require_accel = args.require_accelerator or bool(base.get("require_accelerator", False))
    out_dir = str(base.get("output_dir", "data/results"))

    device = get_device(device_pref, require_accelerator=require_accel)
    print(f"[score] resolved device = {device}")

    # dataset/model naming
    name = {"mnist":"mnist","fashion_mnist":"fashion","cifar10":"cifar10"}[args.dataset]
    model_key = args.model

    # checkpoint path
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_name = f"{name}_{model_key}.pt"
    ckpt_path = ckpt_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load dataset (test split) & model
    _, ds_test = load_dataset(args.dataset)
    # Your model loading here (pseudo: you likely have a builder that can load_state_dict)
    model_ctor = MODEL_REGISTRY[args.model]
    model = model_ctor(num_classes=10) if callable(model_ctor) else model_ctor
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state.get("state_dict", state))
    model.to(device).eval()

    # compute per-sample metrics (pseudo-outline)
    ids, y_true, conf, ent, grad = [], [], [], [], []
    # ... iterate DataLoader over ds_test computing logits, entropy, grad-norm ...
    # (omitted for brevity since you already have this implemented)

    # normalize & hybrid
    ent_n = minmax_norm(ent)
    grad_n = minmax_norm(grad)
    hybrid = hybrid_score(ent_n, grad_n)

    # save CSV
    out_csv = Path(out_dir) / f"scores_{name}_{model_key}.csv"
    df = pd.DataFrame({
        "id": ids,
        "y_true": y_true,
        "confidence": conf,
        "entropy": ent,
        "grad_norm": grad,
        "entropy_norm": ent_n,
        "grad_norm_norm": grad_n,
        "hybrid": hybrid,
    })
    df.to_csv(out_csv, index=False)
    print(f"[score] saved: {out_csv}  (N={len(df)})")

    # --- (3) Stamp resolved device into a small sidecar metadata JSON
    meta = {
        "dataset": name,
        "model": model_key,
        "resolved_device": str(device),
    }
    out_meta = Path(out_dir) / f"scores_{name}_{model_key}.meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[score] saved meta: {out_meta}")

if __name__ == "__main__":
    main()
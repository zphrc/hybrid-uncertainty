# ============================================================
#
# Entry point for model training. Loads datasets and lightweight
# CNN architectures, configures the optimizer and training loop,
# and saves best-performing model checkpoints for uncertainty
# analysis in later stages.
#
# ============================================================

#!/usr/bin/env python
import argparse
from pathlib import Path
import yaml
import torch

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.data.datasets import load_dataset, split_train
from src.models import shufflenet_v2, mobilenet_v3_small, efficientnet_v2_s
from src.train.trainer import train

MODEL_REGISTRY = {
    "shufflenet_v2_0_5": shufflenet_v2.build_0_5 if hasattr(shufflenet_v2, "build_0_5") else shufflenet_v2.ShuffleNetV2_0_5,
    "mobilenet_v3_small": mobilenet_v3_small.build if hasattr(mobilenet_v3_small, "build") else mobilenet_v3_small.MobileNetV3Small,
    "efficientnet_v2_s": efficientnet_v2_s.build if hasattr(efficientnet_v2_s, "build") else efficientnet_v2_s.EfficientNetV2S,
}

def parse_args():
    p = argparse.ArgumentParser("hu-train")
    p.add_argument("--dataset", required=True, choices=["mnist","fashion_mnist","cifar10"])
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--device-preference", default=None, choices=["auto","mps","cuda","cpu"],
                   help="Override config.device_preference")
    p.add_argument("--require-accelerator", action="store_true",
                   help="Fail if no GPU/MPS (overrides config.require_accelerator)")
    return p.parse_args()

def main():
    args = parse_args()

    # --- load base config
    with open(args.config) as f:
        base = yaml.safe_load(f) or {}

    device_pref = args.device_preference or base.get("device_preference", "auto")
    require_accel = args.require_accelerator or bool(base.get("require_accelerator", False))
    seed = int(base.get("seed", 42))
    batch_size = int(base.get("batch_size", 64))
    epochs = int(base.get("epochs", 15))
    lr = float(base.get("lr", 1e-3))
    out_dir = str(base.get("output_dir", "data/results"))

    # --- resolve device
    device = get_device(device_pref, require_accelerator=require_accel)
    print(f"[train] resolved device = {device}")

    # --- seed
    try:
        set_seed(seed)
    except Exception:
        pass

    # --- data
    ds_train_full, ds_test = load_dataset(args.dataset)
    ds_train, ds_val = split_train(ds_train_full, split_ratio=0.8)

    # --- model
    model_ctor = MODEL_REGISTRY[args.model]
    model = model_ctor(num_classes=10) if "cifar" not in args.dataset else model_ctor(num_classes=10)

    # --- name / checkpoint
    name = {"mnist":"mnist","fashion_mnist":"fashion","cifar10":"cifar10"}[args.dataset]
    model_key = args.model
    ckpt_name = f"{name}_{model_key}.pt"
    Path(out_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

    # --- train
    train(
        model=model,
        device=device,
        train_set=ds_train,
        val_set=ds_val,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        out_dir=out_dir,
        ckpt_name=ckpt_name,
    )

if __name__ == "__main__":
    main()

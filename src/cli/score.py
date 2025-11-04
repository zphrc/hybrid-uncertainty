import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.transforms import get_transforms
from src.data.datasets import load_dataset
from src.models.shufflenet_v2 import shufflenet_v2_0_5
from src.models.mobilenet_v3_small import mobilenet_v3_small
from src.models.efficientnet_v2_s import efficientnet_v2_s
from src.uncertainty.entropy import softmax_entropy
from src.uncertainty.gradients import grad_sensitivity

MODEL_FN = {
    "shufflenet_v2_0_5": shufflenet_v2_0_5,
    "mobilenet_v3_small": mobilenet_v3_small,
    "efficientnet_v2_s": efficientnet_v2_s,
}

def parse_args():
    p = argparse.ArgumentParser(description="Compute per-sample entropy, gradient sensitivity, and hybrid danger score")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--model", type=str, default="shufflenet_v2_0_5",
                   choices=list(MODEL_FN.keys()))
    p.add_argument("--ckpt", type=str, default="", help="Path to checkpoint .pt (if empty, use default path)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--out", type=str, default="data/results")
    return p.parse_args()

def main():
    args = parse_args()

    base = yaml.safe_load(open("config/base.yaml"))
    ds_cfg = yaml.safe_load(open("config/datasets.yaml"))
    set_seed(base["seed"])
    device = get_device(base.get("device_preference", "mps"))

    # Dataset settings
    if args.dataset == "mnist":
        dname = "MNIST"; grayscale = True; num_classes = 10
        ckpt_default = "mnist_shufflenet_v2_0_5.pt"
    elif args.dataset in ["fashion_mnist", "fashionmnist"]:
        dname = "FashionMNIST"; grayscale = True; num_classes = 10
        ckpt_default = "fashion_mnist_shufflenet_v2_0_5.pt"
    else:
        dname = "CIFAR10"; grayscale = False; num_classes = 10
        ckpt_default = "cifar10_mobilenet_v3_small.pt" if args.model == "mobilenet_v3_small" else "cifar10_efficientnet_v2_s.pt"

    tfm = get_transforms(grayscale=grayscale)
    _, ds_test = load_dataset(dname, ds_cfg[args.dataset]["root"], ds_cfg[args.dataset]["download"], tfm)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size or base["batch_size"], shuffle=False)

    # Build model & load checkpoint
    model = MODEL_FN[args.model](num_classes=num_classes, grayscale=grayscale if args.model!="efficientnet_v2_s" else False)
    model.to(device).eval()

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(base.get("output_dir", "data/results")) / "checkpoints" / ckpt_default
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])

    # Compute scores
    all_rows = []
    with torch.no_grad():
        pass  # we'll enable grad during gradient sensitivity

    # We'll do two passes: one to collect raw entropy and grad norms,
    # then normalize grad norms across the whole test set.
    ent_list, grad_list, y_true_list, y_pred_list, conf_list = [], [], [], [], []
    idx_counter = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.numpy()  # keep labels on CPU numpy for saving
        # logits & entropy
        x.requires_grad_(False)
        logits = model(x)
        entropy = softmax_entropy(logits)  # [B]
        # gradient sensitivity (requires grad)
        grad_norms = compute_grad_norms(model, x)  # [B]

        # predictions & top-1 conf
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)  # top-1 prob and idx

        ent_list.append(entropy.detach().cpu().numpy())
        grad_list.append(grad_norms.detach().cpu().numpy())
        y_true_list.append(y)
        y_pred_list.append(pred.detach().cpu().numpy())
        conf_list.append(conf.detach().cpu().numpy())
        idx_counter += x.size(0)

    ent = np.concatenate(ent_list)
    grad = np.concatenate(grad_list)
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    conf = np.concatenate(conf_list)

    # Normalize
    ent_norm = ent / np.log(float(num_classes))  # entropy in [0,1]
    gmin, gmax = float(grad.min()), float(grad.max())
    grad_norm = (grad - gmin) / (gmax - gmin + 1e-12)  # [0,1]

    hybrid = 0.5 * (ent_norm + grad_norm)

    # Save CSV
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"scores_{args.dataset}_{args.model}.csv"
    df = pd.DataFrame({
        "entropy_raw": ent,
        "entropy_norm": ent_norm,
        "grad_norm_raw": grad,
        "grad_norm_norm": grad_norm,
        "hybrid": hybrid,
        "y_true": y_true,
        "y_pred": y_pred,
        "top1_conf": conf
    })
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}  (N={len(df)})")
    print(f"grad min/max: {gmin:.4f} / {gmax:.4f}")

def compute_grad_norms(model, x_batch: torch.Tensor) -> torch.Tensor:
    """
    L2 norm of d(logit_{pred})/dx per sample.
    Must run with gradients enabled.
    """
    model.eval()
    x = x_batch.detach().clone().requires_grad_(True)
    logits = model(x)
    target = logits.argmax(dim=1)
    out = logits[torch.arange(x.size(0), device=x.device), target]
    model.zero_grad(set_to_none=True)
    out.sum().backward()
    g = x.grad.view(x.size(0), -1)
    return g.norm(p=2, dim=1)

if __name__ == "__main__":
    main()

import argparse
import yaml
from pathlib import Path
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.transforms import get_transforms
from src.data.datasets import load_dataset, split_train
from src.models.shufflenet_v2 import shufflenet_v2_0_5
from src.models.mobilenet_v3_small import mobilenet_v3_small
from src.models.efficientnet_v2_s import efficientnet_v2_s
from src.train.trainer import train

MODEL_FN = {
    "shufflenet_v2_0_5": shufflenet_v2_0_5,
    "mobilenet_v3_small": mobilenet_v3_small,
    "efficientnet_v2_s": efficientnet_v2_s,
}

def parse_args():
    p = argparse.ArgumentParser(description="Train lightweight CNNs for hybrid uncertainty evaluation")
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fashion_mnist", "cifar10"],
                   help="Dataset to train on")
    p.add_argument("--model", type=str, default="shufflenet_v2_0_5",
                   choices=list(MODEL_FN.keys()),
                   help="Model architecture to train")
    return p.parse_args()

def main():
    args = parse_args()

    base = yaml.safe_load(open("config/base.yaml"))
    ds_cfg = yaml.safe_load(open("config/datasets.yaml"))

    set_seed(base["seed"])
    device = get_device(base["device_preference"])

    # --- Dataset setup ---
    d = ds_cfg[args.dataset]
    tfm = get_transforms(grayscale=d["grayscale"])
    ds_train_full, ds_test = load_dataset(d["name"], d["root"], d["download"], tfm)
    ds_train, ds_val = split_train(ds_train_full, d["split"]["train"])

    # --- Model setup ---
    grayscale = d["grayscale"]
    model = MODEL_FN[args.model](num_classes=10, grayscale=grayscale)

    # --- Checkpoint name ---
    ckpt_name = f"{args.dataset}_{args.model}.pt"

    # --- Train ---
    train(
        model,
        device,
        ds_train,
        ds_val,
        epochs=base["epochs"],
        lr=base["lr"],
        batch_size=base["batch_size"],
        out_dir=base["output_dir"],
        ckpt_name=ckpt_name,
    )

if __name__ == "__main__":
    main()

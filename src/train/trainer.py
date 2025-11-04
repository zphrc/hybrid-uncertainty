from torch.utils.data import DataLoader
import torch, torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

def train(
    model,
    device,
    train_set,
    val_set,
    epochs=10,
    lr=1e-3,
    batch_size=64,
    out_dir: str = "data/results",
    ckpt_name: str = "mnist_shufflenet_v2_0_5.pt",
):        
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    best_val_acc = -1.0
    out_path = Path(out_dir) / "checkpoints"
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = out_path / ckpt_name

    for epoch in range(1, epochs+1):
        model.train()
        for x,y in tqdm(train_loader, desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        # quick val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct/total if total else 0.0
        print(f"val acc: {acc:.3f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": acc,
                },
                ckpt_file,
            )
            print(f"[saved] {ckpt_file} (val acc {acc:.3f})")

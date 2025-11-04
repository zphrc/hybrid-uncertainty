# Hybrid Uncertainty (Entropy + Gradients)

Implements the thesis pipeline: softmax entropy + input gradient sensitivity → normalized → Hybrid Danger Score → thresholding → AUROC/ECE/AvUC/ARCs. Matches Chapter 4 design. :contentReference[oaicite:3]{index=3}

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hu-train

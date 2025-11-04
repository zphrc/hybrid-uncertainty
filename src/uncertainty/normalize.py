import torch
def normalize_entropy(entropy: torch.Tensor, num_classes: int) -> torch.Tensor:
    return entropy / torch.log(torch.tensor(float(num_classes)))

def normalize_linear(x: torch.Tensor) -> torch.Tensor:
    # min-max normalize per-batch
    xmin, xmax = x.min(), x.max()
    if (xmax - xmin) < 1e-12:
        return torch.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

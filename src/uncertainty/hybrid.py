import torch
from .normalize import normalize_entropy, normalize_linear

def hybrid_danger_score(entropy, grad_norms, num_classes: int) -> torch.Tensor:
    e_hat = normalize_entropy(entropy, num_classes)
    g_hat = normalize_linear(grad_norms)
    return 0.5 * (e_hat + g_hat)

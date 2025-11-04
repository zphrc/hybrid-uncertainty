import torch

def get_device(preference: str = "mps") -> torch.device:
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

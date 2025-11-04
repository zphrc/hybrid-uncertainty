# ============================================================
#
# Detects and manages the available computation device for
# experiments. Prioritizes Apple MPS, CUDA, or CPU based on
# system availability to ensure optimized performance across
# training and evaluation stages.
#
# ============================================================

import os
import torch

ACCELS = ("mps", "cuda")

def _has_mps() -> bool:
    return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

def _has_cuda() -> bool:
    return torch.cuda.is_available()

def get_device(preference: str = "auto", require_accelerator: bool = False) -> torch.device:
    """
    Resolve a torch.device based on preference and availability.

    preference: "auto" | "mps" | "cuda" | "cpu"
    require_accelerator: if True, raise if we would end up on CPU.
    Environment override: HU_DEVICE=[mps|cuda|cpu|auto]
    """
    # 1) env var override (highest priority)
    env_pref = os.environ.get("HU_DEVICE", "").strip().lower()
    if env_pref:
        preference = env_pref

    pref = (preference or "auto").lower()

    # 2) explicit choices
    if pref == "mps":
        if _has_mps(): 
            print("[device] using MPS (Apple Metal)")
            return torch.device("mps")
        raise RuntimeError("Requested MPS but it's not available.")
    if pref == "cuda":
        if _has_cuda():
            print(f"[device] using CUDA (GPU count={torch.cuda.device_count()})")
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA but it's not available.")
    if pref == "cpu":
        print("[device] using CPU (explicit)")
        return torch.device("cpu")

    # 3) auto mode: prefer MPS → CUDA → CPU
    if pref == "auto":
        if _has_mps():
            print("[device] auto-selected MPS")
            return torch.device("mps")
        if _has_cuda():
            print(f"[device] auto-selected CUDA (GPU count={torch.cuda.device_count()})")
            return torch.device("cuda")
        # fallback
        msg = "[device] auto-selected CPU (no accelerator detected)"
        if require_accelerator:
            raise RuntimeError(msg + " and require_accelerator=True")
        print(msg)
        return torch.device("cpu")

    # 4) unknown preference
    raise ValueError(f"Unknown device preference: {preference}")

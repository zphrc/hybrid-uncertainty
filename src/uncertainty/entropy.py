# ============================================================
#
# Computes softmax entropy to quantify aleatoric uncertainty
# in model predictions. Used as the data-driven component of
# the hybrid danger score to evaluate prediction ambiguity.
#
# ============================================================

import torch, torch.nn.functional as F

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)  # nats
    return entropy

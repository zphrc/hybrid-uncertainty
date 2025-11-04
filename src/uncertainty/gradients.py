# ============================================================
#
# Calculates input gradient norms to estimate epistemic
# uncertainty in trained models. Captures model sensitivity
# to input perturbations and complements entropy-based metrics
# in the hybrid uncertainty evaluation framework.
#
# ============================================================

import torch

@torch.no_grad()

def grad_sensitivity(model, x, target_idx=None, use_logit=True):
    """
    L2 norm of d(output)/dx per sample.
    If target_idx is None, uses predicted class.
    """
    model.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_(True)
    logits = model(x)
    if target_idx is None:
        target_idx = logits.argmax(dim=1)
    out = logits[torch.arange(x.size(0)), target_idx]
    out.sum().backward()
    g = x.grad.view(x.size(0), -1)
    return g.norm(p=2, dim=1)  # per-sample gradient norm

# ============================================================
#
# Sets and controls random seeds for reproducibility across
# all experiments. Ensures consistent data shuffling, weight
# initialization, and result replication between runs.
#
# ============================================================

import random, os, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

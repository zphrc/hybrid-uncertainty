# ============================================================
#
# Implements the EfficientNetV2-S lightweight CNN architecture,
# emphasizing accuracy through compound scaling and progressive
# learning. Used as the accuracy-oriented model for uncertainty
# evaluation on CIFAR-10 experiments.
#
# ============================================================

import torchvision.models as tv
import torch.nn as nn

def efficientnet_v2_s(num_classes: int = 10):
    m = tv.efficientnet_v2_s(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

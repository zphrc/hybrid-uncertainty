# ============================================================
#
# Defines the ShuffleNetV2 (0.5×) lightweight CNN architecture
# used for efficiency-focused image classification experiments.
# Adapted for grayscale inputs and configured for uncertainty
# evaluation on MNIST and Fashion-MNIST datasets.
#
# ============================================================

import torch.nn as nn
import torchvision.models as tv

def shufflenet_v2_0_5(num_classes: int = 10, grayscale: bool = True):
    m = tv.shufflenet_v2_x0_5(weights=None)
    if grayscale:
        m.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

import torch.nn as nn
import torchvision.models as tv

def mobilenet_v3_small(num_classes: int = 10, grayscale: bool = False):
    m = tv.mobilenet_v3_small(weights=None)
    if grayscale:
        first = m.features[0][0]  # Conv2d
        m.features[0][0] = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size,
                                     stride=first.stride, padding=first.padding, bias=False)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

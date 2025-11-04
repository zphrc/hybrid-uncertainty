# ============================================================
#
# Defines preprocessing transformations applied to all datasets
# prior to model training and evaluation. Includes tensor
# conversion and normalization functions for both grayscale and
# RGB image data to ensure consistency and stable convergence
# across experiments.
#
# ============================================================

from torchvision import transforms

def get_transforms(grayscale: bool):
    if grayscale:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST/Fashion defaults
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2470, 0.2435, 0.2616)),  # CIFAR-10
        ])

import torch

def np2th(weights, conv=False):
    """Convert HWIO to OIHW"""

    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
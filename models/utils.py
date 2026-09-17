import torch.nn.functional as F


def l2_normalize(x, dim=1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

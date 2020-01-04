import torch
import numpy as np


def Cutout(x,n_holes,length):
    d = x.size(2)
    h = x.size(3)
    w = x.size(4)

    mask = np.ones((d,h, w), np.float32)

    for n in range(n_holes):
        mask_d = np.random.randint(d-length)
        mask_h = np.random.randint(h-length)
        mask_w = np.random.randint(w-length)

        mask[mask_d:mask_d+length, mask_h:mask_h+length, mask_w:mask_w+length] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(x)
    x = x * mask
    return x
import numpy as np
import torch as torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


def color_correction(img, p: int, m: int):
    """

    Parameters
    ----------
    img             Input image
    p               Minkowski Norm order (6 recommended )
    m               Normalizaion order (0, 1, 2)

    Returns
    -------
    out             Normalized Image

    """
    img = np.asarray(img)
    src1 = img.astype(float)
    src1_pow_p = src1 ** p
    n_elements = src1_pow_p[:, :, 0].size

    color_mean = (np.sum(src1_pow_p, axis=(0, 1)) / n_elements) ** (1 / p)

    if m == 1:
        r = np.max(color_mean)
        normalization = (color_mean ** -1) * r
    elif m == 2:
        r = np.sqrt(np.sum(color_mean ** 2))
        normalized = color_mean / r
        r = np.max(normalized)
        normalization = (normalized ** -1) * r
    else:
        # m == 0:
        r = np.sum(color_mean) / 3
        normalization = (color_mean ** -1) * r

    out = normalization * src1
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return Image.fromarray(out)


def color_correction_torch(img, p: int, m: int):
    """

    Parameters
    ----------
    img             Input image as tensor torch
    p               Minkowski Norm order (6 recommended )
    m               Normalizaion order (0, 1, 2)

    Returns
    -------
    out             Normalized Image

    """
    img = torch.from_numpy(np.asarray(img))
    src1 = img.double()
    src1_pow_p = src1 ** p
    n_elements = src1_pow_p[:, :, 0].numel()

    color_mean = (torch.sum(src1_pow_p, axis=(0, 1)) / n_elements) ** (1 / p)

    if m == 1:
        r = torch.max(color_mean)
        normalization = (color_mean ** -1) * r
    elif m == 2:
        r = torch.sqrt(torch.sum(color_mean ** 2))
        normalized = color_mean / r
        r = torch.max(normalized)
        normalization = (normalized ** -1) * r
    else:
        # m == 0:
        r = torch.sum(color_mean) / 3
        normalization = (color_mean ** -1) * r

    out = normalization * src1
    out = torch.clamp(out, 0, 255)
    out = out.to(torch.uint8)

    return Image.fromarray(out.numpy())

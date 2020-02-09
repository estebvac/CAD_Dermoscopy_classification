import numpy as np
import torch as torch
from PIL import Image
from torchvision.transforms import ToTensor


def normalize_staining(img: np.ndarray, io=240, alpha=1, beta=0.15):
    """
    Normalize staining appearence of H&E stained images,
    Original repository:
    https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

    Input:
        img: BGR input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    """
    img = np.asarray(img)

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]
    if len(ODhat) != 0:
        cov = np.cov(ODhat.T)
    else:
        img = img.reshape((h, w, c))
        return img

    if not np.isnan(cov).any() and \
            not np.isinf(cov).any():
        # compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)
    else:
        img = img.reshape((h, w, c))
        return img

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Image.fromarray(Inorm)


def normalize_staining_torch(img: torch.tensor, io=240, alpha=1, beta=0.15):
    """
    Normalize staining appearence of H&E stained images,
    Original repository:
    https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

    Input:
        I: RGB input image as tensor
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    """
    HERef = torch.tensor([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    HERef = HERef.to(device)

    maxCRef = torch.tensor([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -torch.log((img.double() + 1) / io)

    # remove transparent pixels
    ODhat = OD[~torch.any(OD < beta, axis=1)]
    if len(ODhat) != 0:
        cov = covar(ODhat.T)
    else:
        img = img.reshape((h, w, c))
        return img

    if not isnan(cov).any() and \
            not torch.isinf(cov).any():
        # compute eigenvectors
        eigvals, eigvecs = torch.symeig(cov, eigenvectors=True)
    else:
        img = img.reshape((h, w, c))
        return img

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.mm(eigvecs[:, 1:3])
    phi = torch.atan2(That[:, 1], That[:, 0])

    minPhi = percentile(phi, alpha)
    maxPhi = percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].mm(torch.tensor([[torch.cos(minPhi), torch.sin(minPhi)]]).T.to(device))
    vMax = eigvecs[:, 1:3].mm(torch.tensor([[torch.cos(maxPhi), torch.sin(maxPhi)]]).T.to(device))

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        # HE = torch.array((vMin[:, 0], vMax[:, 0])).T
        HE = torch.cat((vMin[:, 0].reshape(1, -1), vMax[:, 0].reshape(1, -1)), 0).T
    else:
        # HE = np.array((vMax[:, 0], vMin[:, 0])).T
        HE = torch.cat((vMax[:, 0].reshape(1, -1), vMin[:, 0].reshape(1, -1)), 0).T

    # rows correspond to channels (RGB), columns to OD values
    Y = torch.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = torch.pinverse(HE).mm(Y)

    # normalize stain concentrations
    maxC = torch.tensor([percentile(C[0, :], 99), percentile(C[1, :], 99)])
    tmp = torch.div(maxC, maxCRef)
    C2 = torch.div(C, tmp[:, np.newaxis].to(device))

    # recreate the image using reference mixing matrix
    Inorm = io * torch.exp(-HERef.double().mm(C2))
    Inorm[Inorm > 255] = 254
    Inorm = torch.reshape(Inorm.T, (h, w, 3)).to(torch.uint8)

    return Inorm


def covar(m, y=None):
    """

    Parameters
    ----------
    m       tesor 1 to compute covariance
    y       tesor 2 to compute covariance

    Returns
    -------
    Covariance of the same tensor if only 1 tesor specified
    Covariance of the 2 tensors if only 2 tesors specified

    """
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def isnan(x):
    return x != x


def percentile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return torch.tensor(result).double()

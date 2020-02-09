from preprocessing.color_correction import color_correction_torch


class ColorConstancyTransform:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.p = 6  # Minkowski Norm order (6 recommended )
        self.m = 1  # Normalizaion order (0, 1, 2)

    def __call__(self, x):
        return color_correction_torch(x, self.p, self.m)

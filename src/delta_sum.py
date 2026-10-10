import numpy as np


def delta_sum(c, delta, fe, ek, pairing):
    """
    :param C: constant containing all mutliplictive factors, normalisations and 
    the interaction potential U
    :param Delta: single valued gap fucntion
    :param fe: fermi-dirac distirbution at temperature T
    :param Ek: eigen energy (2x2 matrix its simply as given, however nxn 
    required direct diagonalisation)
    :param pairing: pairing symmetry definign gap type ie cos(kx)-cos(ky)
    :return: single valued gap function in a self-consistent style
    """
    return c * np.sum(delta * pairing * (1 - 2 * fe) / ek)

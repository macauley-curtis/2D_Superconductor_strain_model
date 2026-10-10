import numpy as np


def fermi_dirac_func(e, kt):
    """
    :param e: energy, same units as kt
    :param kt: botlzman constant * temperature, same units as e
    :return: the fermi-dirac distribution of the energy
    """

    return 1 / (np.exp(e / kt) + 1)

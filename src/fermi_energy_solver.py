import numpy as np


def fermi_energy_solver(e, Ne, gamma_k, KbT):
    """
    Use this to solve for the change in fermi level st it is kept constant 
    with strain"
    It is derived from rearrange the equation for Number of electrons under
    the assumption this is kept constant.
    
    :param e: inputs the energyband
    :return: returns the adjusted fermi level to keep electron number 
    constant with strain
    """
    fermi_energy = Ne / 2 - np.sum(
        1 / (np.exp((e + gamma_k) / KbT) + 1)
    )
    return fermi_energy

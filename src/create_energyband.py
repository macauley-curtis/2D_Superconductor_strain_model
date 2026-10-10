import numpy as np
from scipy.optimize import fsolve

from src.fermi_dirac import fermi_dirac_func
from src.fermi_energy_solver import fermi_energy_solver


def create_energyband(
    strain,
    n_k,
    kx,
    ky,
    t_x,
    t_prime,
    e_0,
    hopping_step,
    hopping_array,
    KbT=1e-6,
):

    """
    Create the energy band for a given strain and hopping parameters.
    
    :param strain: Type of strain (e.g., 'uniaxial', 'c_axis', 'shear').
    :param n_k: Number of k points in each direction (k-space grid will be n_k 
    x n_k).
    :param kx: kx values for the k-space grid
    :param ky: ky values for the k-space grid
    :param t_x: Hopping parameter in the x direction.
    :param t_prime: Next-nearest neighbor hopping parameter.
    :param e_0: Base energy level.
    :param hopping_step: Number of hopping steps to consider.
    :param hopping_array: Array of hopping parameters for each step.
    :param KbT: Thermal energy (default is 1e-6).
    
    
    :return: Energy band array of shape (hopping_step, n_k, n_k
    
    """
    
    # Create the k-space grid and output arrays.
    epsilon_k = np.zeros((hopping_step, n_k, n_k))
    epsilon_0 = np.zeros(hopping_step)

    cos_kx = np.cos(kx)
    cos_ky = np.cos(ky)


    e_k_ne = (
        e_0
        - 2 * t_x * (cos_kx + cos_ky)
        - 4 * t_prime * cos_kx * cos_ky
    )
    Ne = 2 * np.sum(fermi_dirac_func(e_k_ne, KbT))

    hopping_1 = hopping_array[:, 0, None, None]
    hopping_2 = hopping_array[:, 1, None, None]
    
    if strain == "uniaxial":
        gamma_k = (
            -2 * (hopping_1 * cos_kx + hopping_2 * cos_ky)
            - 4 * t_prime * cos_kx * cos_ky
        )
    elif strain == "c_axis":
        gamma_k = (
            -2 * hopping_1 * (cos_kx + cos_ky)
            - 4 * hopping_2 * cos_kx * cos_ky
        )
    elif strain == "shear":
        a1x = np.ones((hopping_step, 1, 1))
        a1y = np.linspace(0, 0.001, hopping_step)[:, None, None]
        a_minus = a1x - a1y
        a_plus = a1x + a1y
        gamma_k = (
            -2 * t_x * (
                np.cos(a1x * kx + a1y * ky)
                + np.cos(a1y * kx + a1x * ky)
            )
            - 2 * hopping_1 * np.cos(a_minus * (kx - ky))
            - 2 * hopping_2 * np.cos(a_plus * (kx + ky))
        )

    # Shift each strained band to preserve the initial electron number.
    for h in range(hopping_step):
        initial_guess = e_0 if h == 0 else epsilon_0[h - 1]
        epsilon_0[h] = fsolve(
            fermi_energy_solver,
            initial_guess,
            args=(Ne, gamma_k[h], KbT),
        )[0]
        epsilon_k[h] = epsilon_0[h] + gamma_k[h]
    return epsilon_k
    
    
        
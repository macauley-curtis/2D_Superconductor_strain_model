import numpy as np

from src.dirac_delta import delta


def dos(
    gap_pairing,
    e_k,
    epsilon,
    dos_step,
    screening_energy,
    hopping_step,
    nk,
    n_components
):
    
    """
    Calculate the density of states (DOS) for a given energy band and gap 
    pairing.
    
    :param gap_pairing: Gap pairing function (can be one-component or 
    two-component).
    :param e_k: Energy band array of shape (hopping_step, nk, nk)
    :param epsilon: Energy values for which to calculate the DOS.
    :param dos_step: Number of energy steps for the DOS calculation.
    :param screening_energy: Screening energy parameter for the delta function.
    :param hopping_step: Number of hopping steps considered in the energy band.
    :param nk: Number of k points in each direction (k-space grid is nk x
    nk).
    :param n_components: Number of components in the gap pairing (1 or 2).
    
    :returns the weighted density of states array of shape 
    (hopping_step, dos_step, n_components)
    
    """
    

    weighted_dos = np.zeros((hopping_step, dos_step, n_components))
    
    inv_k_space_n = 1 / (nk ** 2 * np.pi)
    
    if n_components == 1:
        gap_pairing_sq = gap_pairing ** 2
        
        for hopping in range (hopping_step):
                for e in range(dos_step):
                     weighted_dos[hopping, e] = inv_k_space_n * np.sum(
                         delta(screening_energy, epsilon[e] - e_k[hopping, :, :])
                         * gap_pairing_sq)
        
    else:
        gap_pairing_one_sq = gap_pairing[0] ** 2
        gap_pairing_two_sq = gap_pairing[1] ** 2
        
        for hopping in range (hopping_step):
            for e in range(dos_step):
                weighted_dos[hopping, e, 0] = inv_k_space_n * np.sum(
                    delta(screening_energy, epsilon[e] - e_k[hopping, :, :])
                    * gap_pairing_one_sq)
                
                weighted_dos[hopping, e, 1] = inv_k_space_n * np.sum(
                    delta(screening_energy, epsilon[e] - e_k[hopping, :, :])
                    * gap_pairing_two_sq)
                
    return weighted_dos
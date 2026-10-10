import numpy as np


def get_hopping_range(strain, t_x, t_prime):
    """
    :param strain: type of strain; uniaxial, c_axis or shear
    :param t_x: nearest neighbour unstrained hopping parameter
    :param t_prime: next-nearest neighbour unstrained hopping parameter
    :return: hopping array and corresponding leading parameter
    """
    # From strain returns numeric hopping array and boolean symmetric hopping that decides the type of plot
    hopping_array = np.load(f"hopping_array_{strain}.npy")
    if strain == "shear":
        t_unstrained = t_prime
    else:
        t_unstrained = t_x

    if hopping_array[0] / t_unstrained < 1 and hopping_array[-1] / t_unstrained > 1:
        symmetric_hopping = True
    else:
        symmetric_hopping = False

    return hopping_array, symmetric_hopping, t_unstrained
import numpy as np


def create_hopping_range(
    strain, 
    t_x, 
    t_prime, 
    t_1,
    t_2,
    hopping_step,
    V_xy=0.4):
    """
    :param strain: type of strain; uniaxial, c_axis or shear
    :param t_x: nearest neighbour unstrained hopping parameter
    :param t_prime: next-nearest neighbour unstrained hopping parameter
    :param t_1: lower bound of hopping parameter range
    :param t_2: upper bound of hopping parameter range
    :param hopping_step: number of steps in the hopping parameter range
    :param V_xy: Poisson's ratio for uniaxial strain (default: 0.4)
    :param load_hopping_array: if True, load precomputed hopping array from 
    file; if False, compute hopping array (default: False)
    
    :return: hopping array and corresponding leading parameter
    
    """

    
  
    "calculate hopping changes based on type of strain"
    
    if strain == "c_axis":
        # t_1_array is nearest neighbour hopping parameter range
        # t_2_array is next-nearest neighbour hopping parameter range
        t_1_array = np.linspace(t_1 * t_x, t_2 * t_x, hopping_step)
        t_2_array = (t_1_array[:] - t_x) + t_prime

    if strain == "uniaxial":
        # t_1_array is x nearest neighbour hopping parameter range
        # t_2_array is y nearest neighbour hopping parameter range
        t_1_array = np.linspace(
            t_1 * t_x, t_2 * t_x, hopping_step
        )  # Hopping integral linear change array
        t_2_array = (V_xy * (t_1_array[:] - t_x)) + t_x

    if strain == "shear":
        t_1_array = np.linspace(t_1 * t_prime, t_2 * t_prime, hopping_step)
        t_2_array = t_prime - (0.4 * (t_1_array[:] - t_prime))
        # a_Max = 1
        # a_minus = 0.001
        # a1x = np.linspace(1, a_Max, hopping_step)
        # a1y = np.linspace(0, a_minus, hopping_step)
        
    hopping_array = np.column_stack((t_1_array, t_2_array))
        
    return hopping_array
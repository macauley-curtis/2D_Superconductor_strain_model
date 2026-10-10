import numpy as np


def create_square_k_space(nk, BZ_edge):
    """
    :param nk: number of k space points such that nkx, nky = nk, nk
    :param BZ_edge: set the k-space edge, ie for unstrained square lattice, 
    1st BZ is  -pi, pi so BZ =1
    :return: kx_array,ky_array - nkxnk grids of kx and ky to be operated on 
    array like, and k_x, k_y: lists in k-space
    
    """
    kx = np.zeros((nk, nk))
    ky = np.zeros(np.shape(kx))

    kx_array = np.linspace(-BZ_edge * np.pi, +BZ_edge * np.pi, nk)
    ky_array = np.linspace(+BZ_edge * np.pi, -BZ_edge * np.pi, nk)
    for i in range(nk):
        kx[i, :] = kx_array[:]
        ky[:, i] = ky_array[:]
    return kx, ky

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"Define functions"


def delta(de, x):
    return de / ((x ** 2) + (de ** 2))


def gap_symmetry(gap_pairing, kx, ky):
    "Returns the gap pairing(s)"
    if gap_pairing == "s_wave":
        return 1
    if gap_pairing == "d_wave":
        return (np.cos(kx[:, :]) - np.cos(ky[:, :]))
    if gap_pairing == "ext_s_wave":
        return np.cos(kx[:, :]) + np.cos(ky[:, :])
    if gap_pairing == "d_plus_id":
        return np.cos(kx[:, :]) - np.cos(ky[:, :]),  np.sin(kx[:, :]) * np.sin(ky[:, :])
    if gap_pairing == "d_plus_ig":
        return (np.cos(kx[:, :]) - np.cos(ky[:, :])), (np.sin(kx) * np.sin(ky)) * (np.cos(kx[:, :]) - np.cos(ky[:, :]))
    if gap_pairing == "chiral_p":
        return np.sin(kx[:, :]), np.sin(ky[:, :])


def dos(strain, gap_pairing, dos_step, normal_state, fermi_surface):
    StartTime = datetime.now()
    "Load e_k depending on strain type"
    epsilon_k = np.load("Energyband_{}.npy".format(strain))
    hopping_step, n_x, n_y = np.shape(epsilon_k) #Define shape of arrays to load into main calculation


    "Define how many components to model"
    if gap_pairing == "s_wave" or gap_pairing == "ext_s_wave" or gap_pairing == "d_wave":
        gap_type = "one component"
    else:
        gap_type = "two component"

    "Depending on what calculation is called, set and save outputs"
    if gap_type == "one component" and normal_state and fermi_surface:
        var_1, var_2 = one_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k)
        np.save("{}_{}_dos".format(gap_pairing, strain), var_1)
        np.save("{}_{}_fermi_surface".format(gap_pairing, strain), var_2)
    else:
        if gap_type == "one component" and normal_state:
            var = one_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k)
            np.save("{}_{}_dos".format(gap_pairing, strain), var)
        if gap_type == "one component" and fermi_surface:
            var = one_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k)
            np.save("{}_{}_fermi_surface".format(gap_pairing, strain), var)


    if gap_type == "two component" and  normal_state and fermi_surface:
        x_dos, y_dos, fermi_x, fermi_y = two_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k)

        np.save("{}_x_dos_{}".format(gap_pairing, strain), x_dos)
        np.save("{}_y_dos_{}".format(gap_pairing, strain), y_dos)
        np.save("{}_x_fermi_surface_{}".format(gap_pairing, strain), fermi_x)
        np.save("{}_y_fermi_surface_{}".format(gap_pairing, strain), fermi_y)
    else:
        if normal_state:
            x_dos, y_dos = two_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k)
            np.save("{}_x_dos_{}".format(gap_pairing, strain), x_dos)
            np.save("{}_x_dos_{}".format(gap_pairing, strain), y_dos)
        if fermi_surface:
            fermi_x, fermi_y = two_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k)
            np.save("{}_x_fermi_surface_{}".format(gap_pairing, strain), fermi_x)
            np.save("{}_y_fermi_surface_{}".format(gap_pairing, strain), fermi_y)

    EndTime = datetime.now()
    print("Runtime =", EndTime - StartTime)



def one_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k):
    if gap_type == "two component":
        exit()

    " set up k space in first Brilluion zone  "
    k_x = np.linspace(-1 * np.pi, +1 * np.pi, n_x)
    k_y = np.linspace(+1 * np.pi, -1 * np.pi, n_y)
    kx = np.zeros((n_x, n_y))
    ky = np.zeros((np.shape(kx)))

    for i in range(n_x):
        kx[i, :] = k_x[:]
        ky[:, i] = k_y[:]

    "Set physical constants"
    t_x = 81.62 * 10 ** -3  # Hopping integral value unstrained
    de = 1 * 10 ** -3  # screening energy to calculate density of states approximaiton
    inv_k_space = 1 / (
            n_x * n_y * np.pi)  # calculate inverse of k_space multiplied for pi to correctly convert this DOS int. to a DOS sum
    epsilon = np.linspace(-2 * t_x, 2.5 * t_x, dos_step)  # energy range of width of DOS to calculate
    gap_pairing_sq = gap_symmetry(gap_pairing, kx, ky) ** 2
    second_fermi_strain = 16 #strain slice at which to output the second Fermi surface contour

    " Create zero arrays to be written into "
    if normal_state:
        dos_NS_weighted = np.zeros((hopping_step, dos_step))

    "For loops to calculate the DOS for each hopping step, accross the enegry range, calling Delta as a func of temp"
    for hopping in range(hopping_step):
        for e_step in range(dos_step):
            if normal_state:
                dos_NS_weighted[hopping, e_step] = inv_k_space * np.sum(delta(de, epsilon[e_step] - epsilon_k[hopping, :, :]) * gap_pairing_sq)


    "Calculates Fermi surface"
    if fermi_surface:
        dos_nodal = np.zeros((2, n_x, n_y))
        dos_nodal[0, :, :] = delta(de, 0 - epsilon_k[0, :, :]) * gap_pairing_sq
        dos_nodal[1, :, :] = delta(de, 0 - epsilon_k[second_fermi_strain, :, :]) * gap_pairing_sq

    "Output Variables"
    if normal_state and fermi_surface:
        return dos_NS_weighted, dos_nodal
    else:
        if normal_state:
            return dos_NS_weighted
        if fermi_surface:
            return dos_nodal


def two_comp(strain, gap_pairing, gap_type, normal_state, fermi_surface, dos_step, hopping_step, n_x, n_y, epsilon_k):
    if gap_type == "one component":
        exit()
    " set up k space in first Brilluion zone  "
    k_x = np.linspace(-1 * np.pi, +1 * np.pi, n_x)
    k_y = np.linspace(+1 * np.pi, -1 * np.pi, n_y)
    kx = np.zeros((n_x, n_y))
    ky = np.zeros((np.shape(kx)))

    for i in range(n_x):
        kx[i, :] = k_x[:]
        ky[:, i] = k_y[:]

    "Set physical constants"
    t_x = 81.62 * 10 ** -3  # Hopping integral value unstrained
    de = 1 * 10 ** -3  # screening energy to calculate density of states approximaiton
    inv_k_space = 1 / (
            n_x * n_y * np.pi)  # calculate inverse of k_space multiplied for pi to correctly convert this DOS int. to a DOS sum
    epsilon = np.linspace(-2 * t_x, 2.5 * t_x, dos_step)  # energy range of width of DOS to calculate
    gap_pairing_x, gap_pairing_y = gap_symmetry(gap_pairing, kx, ky)
    gap_pairing_x_sq, gap_pairing_y_sq = gap_pairing_x ** 2, gap_pairing_y ** 2

    " Create zero arrays to be written into "
    if normal_state:
        dos_NS_weighted_x = np.zeros((hopping_step, dos_step))
        dos_NS_weighted_y = np.zeros_like(dos_NS_weighted_x)

        "For loops to calculate the DOS for each hopping step, accross the enegry range, calling Delta as a func of temp"
        for hopping in range(hopping_step):
            for e_step in range(dos_step):
                if normal_state:
                    dos_NS_weighted_x[hopping, e_step] = inv_k_space * np.sum(
                        delta(de, epsilon[e_step] - epsilon_k[hopping, :, :]) * gap_pairing_x_sq)
                    dos_NS_weighted_y[hopping, e_step] = inv_k_space * np.sum(
                        delta(de, epsilon[e_step] - epsilon_k[hopping, :, :]) * gap_pairing_y_sq)

    "Calculates Fermi surface"
    if fermi_surface:
        dos_nodal_x = np.zeros((2, n_x, n_y))
        dos_nodal_y = np.zeros((2, n_x, n_y))
        dos_nodal_x[0, :, :] = delta(de, 0 - epsilon_k[0, :, :]) * gap_pairing_x_sq
        dos_nodal_x[1, :, :] = delta(de, 0 - epsilon_k[16, :, :]) * gap_pairing_x_sq
        dos_nodal_y[0, :, :] = delta(de, 0 - epsilon_k[0, :, :]) * gap_pairing_y_sq
        dos_nodal_y[1, :, :] = delta(de, 0 - epsilon_k[16, :, :]) * gap_pairing_y_sq

    "Output variables "
    if normal_state and fermi_surface:
        return dos_NS_weighted_x,dos_NS_weighted_y,dos_nodal_x,dos_nodal_y
    else:
        if normal_state:
            return dos_NS_weighted_x[:, :], dos_NS_weighted_y[:, :]
        if fermi_surface:
            return dos_nodal_x, dos_nodal_y


dos(strain="uniaxial", gap_pairing="s_wave", dos_step=2000, normal_state=True, fermi_surface=True)

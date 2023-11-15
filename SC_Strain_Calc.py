import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from datetime import datetime
import os
import warnings

# Suppresses the runtime warnings due to the fermi-dirac calcultion at temp near 0
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fermi_dirac_func(e, kt):
    """
    :param e: energy, same units as kt
    :param kt: botlzman constant * temperature, same units as e
    :return: the fermi-dirac distribution of the energy
    """

    return 1 / (np.exp(e / kt) + 1)


def delta_sum_func(C, Delta, fe, Ek, pairing):
    """
    :param C: constant containing all mutliplictive factors, normalisations and the interaction potential U
    :param Delta: single valued gap fucntion
    :param fe: fermi-dirac distirbution at temperature T
    :param Ek: eigen energy (2x2 matrix its simply as given, however nxn required direct diagonalisation)
    :param pairing: pairing symmetry definign gap type ie cos(kx)-cos(ky)
    :return: single valued gap function in a self-consistent style
    """
    return C * np.sum(Delta * pairing * (1 - 2 * fe) / Ek)


def delta(de, x):
    """
    :param de: screening energy
    :param x: posisitonal argument, for the dirac function centred on x
    :return: delta dirac approximation
    """
    return de / ((x**2) + (de**2))


def get_hopping_range(strain, t_x, t_prime):
    """
    :param strain: type of strain; uniaxial, c_axis or shear
    :param t_x: nearest neighbour unstrained hopping parameter
    :param t_prime: next-nearest neighbour unstrained hopping parameter
    :return: hopping array and corresponding leading parameter
    """
    # From strain returns numeric hopping array and boolean symmetric hopping that decides the type of plot
    hopping_array = np.load("hopping_array_{}.npy".format(strain))
    if strain == "shear":
        t_unstrained = t_prime
    else:
        t_unstrained = t_x

    if hopping_array[0] / t_unstrained < 1 and hopping_array[-1] / t_unstrained > 1:
        symmetric_hopping = True
    else:
        symmetric_hopping = False

    return hopping_array, symmetric_hopping, t_unstrained


def gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge):
    """
    :param gap_pairing: type of pairing; s_wave, d_wave, ext_s_wave, chiral_p, d_plus_id, d_plus_ig
    :param strain: type of strain; uniaxial, c_axis or shear
    :param a1x: x component of lattice coordinate a1
    :param a1y: y component of lattice coordinate a1
    :param nk: number of k points, such that k-space grid will be nk x nk
    :param BZ_edge: set the k-space edge, ie for unstrained square lattice, 1st BZ is  -pi, pi so BZ =1
    :return: Interaction potential, gap pairing, titles for plots  specific to pairing/strain combination
    """
    kx, ky, k_x, k_y = k_space(nk, BZ_edge)
    if gap_pairing == "s_wave":
        return -0.0335, 1, "s-wave"
    if gap_pairing == "d_wave":
        if strain == "shear":
            return (
                -0.0394 * 0.5,
                (
                    np.cos(a1x * kx[:, :] + a1y * ky[:, :])
                    - np.cos(a1y * kx[:, :] + a1x * ky[:, :])
                ),
                "d$_{x^2-y^2}$",
            )
        else:
            return -0.0394 * 0.5, np.cos(kx[:, :]) - np.cos(ky[:, :]), "d$_{x^2-y^2}$"

    if gap_pairing == "ext_s_wave":
        if strain == "shear":
            return (
                -0.06483,
                np.cos(a1x * kx[:, :] + a1y * ky[:, :])
                + np.cos(a1y * kx[:, :] + a1x * ky[:, :]),
                "s$^{\pm}$",
            )
        else:
            return -0.06483, np.cos(kx[:, :]) + np.cos(ky[:, :]), "s$^{\pm}$"

    if gap_pairing == "d_plus_id":
        if strain == "shear":
            return (
                -0.5 * 0.03915,
                -0.5 * 0.323549,
                np.cos(kx[:, :]) - np.cos(ky[:, :]),
                np.sin(a1x * kx[:, :] + ky[:, :] * a1y)
                * np.sin(a1x * ky[:, :] + kx[:, :] * a1y),
                "d+id",
                "$d_{x^2-y^2}$",
                "$d_{xy}$",
            )
        else:
            return (
                -0.5 * 0.03915,
                -0.5 * 0.323549,
                np.cos(kx[:, :]) - np.cos(ky[:, :]),
                np.sin(kx[:, :]) * np.sin(ky[:, :]),
                "d+id",
                "$d_{x^2-y^2}$",
                "$d_{xy}$",
            )

    if gap_pairing == "d_plus_ig":
        if strain == "shear":
            return (
                -0.5 * 0.03928,
                -0.5 * 0.496063,
                np.cos(kx[:, :]) - np.cos(ky[:, :]),
                np.sin(a1x * kx[:, :] + ky[:, :] * a1y)
                * np.sin(a1x * ky[:, :] + kx[:, :] * a1y),
                "d+ig",
                "$d_{x^2-y^2}$",
                "g",
            )
        else:
            return (
                -0.5 * 0.03928,
                -0.5 * 0.496063,
                (np.cos(kx[:, :]) - np.cos(ky[:, :])),
                (np.sin(kx) * np.sin(ky)) * (np.cos(kx[:, :]) - np.cos(ky[:, :])),
                "d+ig",
                "$d_{x^2-y^2}$",
                "g",
            )

    if gap_pairing == "chiral_p":
        if strain == "shear":
            return (
                -0.03751,
                -0.03751,
                np.sin(a1x * kx[:, :] + a1y * ky[:, :])
                + np.sin(a1x * ky[:, :] + a1y * kx[:, :]),
                np.sin(a1x * kx[:, :] + a1y * ky[:, :])
                - np.sin(a1x * ky[:, :] + a1y * kx[:, :]),
                "p$_x$ + ip$_y$",
                "p$_x$",
                "p$_y$",
            )
        else:
            return (
                -0.07501,
                -0.07501,
                np.sin(kx[:, :]),
                np.sin(ky[:, :]),
                "p$_x$ + ip$_y$",
                "p$_x$",
                "p$_y$",
            )


def gap_titles(gap_pairing):
    # Returns the interaction potential(s) and the gap pairing(s)
    if gap_pairing == "s_wave":
        return "s-wave", "one component", "Error: One component", "Error: One component"
    if gap_pairing == "d_wave":
        return (
            "d$_{x^2-y^2}$",
            "one component",
            "Error: One component",
            "Error: One component",
        )
    if gap_pairing == "ext_s_wave":
        return (
            "s$^{\pm}$",
            "one component",
            "Error: One component",
            "Error: One component",
        )
    if gap_pairing == "d_plus_id":
        return "d+id", "two component", "$d_{x^2-y^2}$", "$d_{xy}$"
    if gap_pairing == "d_plus_ig":
        return "d+ig", "two component", "$d_{x^2-y^2}$", "g"
    if gap_pairing == "chiral_p":
        return "p$_x$ + ip$_y$", "two component", "p$_x$", "p$_y$"


def round_fermi_vals(strain, hopping_array, fermi_vals):
    # inputs hopping array and fermi vals and outputs the normalised hopping value at each strain specified
    if strain == "shear":
        t_unit = 36.73 * 10**-3
    else:
        t_unit = 81.62 * 10**-3

    val_1, val_2, val_3 = (
        round(hopping_array[fermi_vals[0]] / t_unit, 2),
        round(hopping_array[fermi_vals[1]] / t_unit, 2),
        round(hopping_array[fermi_vals[2]] / t_unit, 2),
    )
    return val_1, val_2, val_3


def k_space(nk, BZ_edge):
    """
    :param nk: number of k space points such that nkx, nky = nk, nk
    :param BZ_edge: set the k-space edge, ie for unstrained square lattice, 1st BZ is  -pi, pi so BZ =1
    :return: kx,ky - nkxnk grids of kx and ky to be operated on array like, and k_x, k_y: lists in k-space
    """
    kx = np.zeros((nk, nk))
    ky = np.zeros((np.shape(kx)))

    k_x = np.linspace(-BZ_edge * np.pi, +BZ_edge * np.pi, nk)
    k_y = np.linspace(+BZ_edge * np.pi, -BZ_edge * np.pi, nk)
    for i in range(nk):
        kx[i, :] = k_x[:]
        ky[:, i] = k_y[:]
    return kx, ky, k_x, k_y


def epsilon_k(
    strain,
    n_k,
    t_1,
    t_2,
    hopping_step,
    t_x,
    t_prime,
    e_0,
    BZ_edge,
    kx,
    ky,
    k_x,
    k_y,
    V_xy,
    plot_nth_contour,
):
    """
    :param plot_nth_contour: choose a slice of the strained array to plot
    :return: one band energyband calculated on the k-space array
    """
    "Use this to solve for the change in fermi level st it is kept constant with strain"
    " It is derived from rearrange the equation for Number of electrons under the assumption this is kept const."

    def fermi_energy_solver(e):
        """
        :param e: inputs the energyband
        :return: returns the adjusted fermi level to keep electron number constant with strain
        """
        return Ne / 2 - np.sum(1 / (np.exp(((e + gamma_k[h, :, :]) / KbT)) + 1))

    "  Set Zero arrays to be written into "
    # gamma_k and epsilon_0 together make up epsilon_k, but separated for constant electron number
    epsilon_k = np.zeros((hopping_step, n_k, n_k))
    gamma_k = np.zeros((np.shape(epsilon_k)))
    epsilon_0 = np.zeros((1, hopping_step))

    " Set physical constants "

    KbT = 0  # Boltzman const * Temperature (if not zero, remember to convert to eV)

    "calculate hopping changes based on type of strain"
    if strain == "c_axis":
        t_x_array = np.linspace(t_1 * t_x, t_2 * t_x, hopping_step)
        t_xy_array = (t_x_array[:] - t_x) + t_prime

    if strain == "uniaxial":
        t_x_array = np.linspace(
            t_1 * t_x, t_2 * t_x, hopping_step
        )  # Hopping integral linear change array
        t_y_array = (V_xy * (t_x_array[:] - t_x)) + t_x

    if strain == "shear":
        t_xy_1 = np.linspace(t_1 * t_prime, t_2 * t_prime, hopping_step)
        t_xy_2 = t_prime - (0.4 * (t_xy_1[:] - t_prime))
        a_Max = 1
        a_minus = 0.001
        a1x = np.linspace(1, a_Max, hopping_step)
        a1y = np.linspace(0, a_minus, hopping_step)

    e_k_ne = (
        e_0
        - (2 * t_x * (np.cos(kx[:, :]) + np.cos(ky[:, :])))
        - 4 * t_prime * (np.cos(kx[:, :]) * np.cos(ky[:, :]))
    )
    Ne = 2 * np.sum(fermi_dirac_func(e_k_ne, KbT))

    " Create an epsilon_k for every combination of (k_x,k_y) points at each strain "
    for h in range(hopping_step):
        if strain == "uniaxial":
            gamma_k[h, :, :] = -2 * (
                t_x_array[h] * np.cos(kx[:, :]) + (t_y_array[h] * np.cos(ky[:, :]))
            ) - 4 * t_prime * (np.cos(kx[:, :]) * np.cos(ky[:, :]))
        if strain == "c_axis":
            gamma_k[h, :, :] = (
                -2 * t_x_array[h] * (np.cos(kx[:, :]) + np.cos(ky[:, :]))
            ) - (4 * t_xy_array[h] * np.cos(kx[:, :]) * np.cos(ky[:, :]))
        if strain == "shear":
            am = a1x[h] - a1y[h]
            ap = a1x[h] + a1y[h]
            gamma_k[h, :, :] = (
                -(
                    2
                    * t_x
                    * (
                        np.cos(a1x[h] * kx[:, :] + a1y[h] * ky[:, :])
                        + np.cos(a1y[h] * kx[:, :] + a1x[h] * ky[:, :])
                    )
                )
                - 2 * (t_xy_1[h] * np.cos(am * (kx[:, :] - ky[:, :])))
                - (2 * t_xy_2[h] * np.cos(ap * (kx[:, :] + ky[:, :])))
            )

        """
         combine above 'gamma' with a strain-solved epsilon_0, using the intial value to calculate the number of electrons
         This keep the number of electrons constant by fixing Ne and readjusting the fermi-level inlcuded in epsilon_0
         This is an assumption that no longer holds if another band is used
        """

        if h == 0:
            epsilon_0[0, h] = fsolve(fermi_energy_solver, e_0)
            epsilon_k[h, :, :] = epsilon_0[0, h] + gamma_k[h, :, :]
        else:
            epsilon_0[0, h] = fsolve(fermi_energy_solver, epsilon_0[0, h - 1])
            epsilon_k[h, :, :] = epsilon_0[0, h] + gamma_k[h, :, :]

    "Plot first and last strains to check it is as expected"
    fig, ax = plt.subplots(figsize=(6, 6))
    first_contour = 0
    cntr1 = ax.contour(k_x, k_y, epsilon_k[first_contour, :, :], colors="red")
    cntr2 = ax.contour(k_x, k_y, epsilon_k[plot_nth_contour, :, :], colors="blue")
    ax.set_xlabel("$k_x$", fontsize=20)
    ax.set_ylabel("$k_y$", fontsize=20)
    ax.set_title(f"$\\varepsilon_k$: {strain}", fontsize=20)
    ax.set_xticks((-BZ_edge * np.pi, 0, BZ_edge * np.pi), ["$-\pi$", "0", "$\pi$"])
    ax.set_yticks((BZ_edge * np.pi, 0, -BZ_edge * np.pi), ["$\pi$", "0", "$-\pi$"])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    h1, _ = cntr1.legend_elements()
    h2, _ = cntr2.legend_elements()
    if strain == "shear":
        number_1 = round(t_xy_1[first_contour] / t_prime, 2)
        number_2 = round(t_xy_1[plot_nth_contour] / t_prime, 2)
        ax.legend(
            [h1[0], h2[0]],
            ["strain ={}".format((number_1)), "strain = {}".format(number_2)],
        )
        return epsilon_k, ax, t_xy_1, t_prime

    else:
        number_1 = round(t_x_array[first_contour] / t_x, 2)
        number_2 = round(t_x_array[plot_nth_contour] / t_x, 2)
        ax.legend(
            [h1[0], h2[0]],
            ["strain ={}".format((number_1)), "strain = {}".format(number_2)],
        )
        return epsilon_k, ax, t_x_array, t_x


def delta_calculation(
    strain,
    gap_type,
    e_k,
    nk,
    Delta_convergence,
    hopping_step,
    temperature_max,
    temperature_step,
    n_k,
    kx,
    ky,
    tolerance,
    mixing,
    Delta_guess,
    gap_pairing,
    a1x,
    a1y,
    BZ_edge,
):
    """
    :return: the gap function as a function of temperature and strain
    """
    " Create k-space to be opperated on array-like"
    k_space_n = n_k**2
    epsilon_k_squared = e_k**2
    "define the temperature range"
    temperature = np.linspace(0, temperature_max, temperature_step)
    boltzman_evt = (
        (1.38 * 10**-23) * temperature / (1.6 * 10**-19)
    )  # Convert into Boltzman eV units for ease

    if gap_type == "one component":
        if gap_type == "two component":
            exit()
        "create zero arrays"
        Delta_Iteration = np.zeros((Delta_convergence, hopping_step))
        Delta_Temperature = np.zeros((hopping_step, temperature_step))
        Tc = np.zeros((hopping_step))

        Interaction_potential, pairing, gap_title = gap_parameters(
            gap_pairing, strain, a1x, a1y, nk, BZ_edge
        )

        " Set Physical Parameters"

        Iteration_Constant = -Interaction_potential / (
            2 * k_space_n
        )  # Combined constant for the integral sum

        "Run calculation"
        for hopping in range(hopping_step):
            if hopping == 0:
                Delta_Iteration[0, 0] = Delta_guess
            else:
                Delta_Iteration[0, 0] = Delta_Temperature[
                    hopping - 1, 0
                ]  # Tc and delta vary smoothly, thus this assumption saves convergence time

            for temp in range(temperature_step):
                for n in range(1, Delta_convergence):
                    delta_pairing = Delta_Iteration[n - 1, 0] * pairing
                    energy_k = np.sqrt(
                        epsilon_k_squared[hopping, :, :] + np.abs(delta_pairing) ** 2
                    )
                    fermi_dirac = 1 / (np.exp(energy_k / boltzman_evt[temp]) + 1)

                    delta_sum = delta_sum_func(
                        Iteration_Constant,
                        delta_pairing,
                        fermi_dirac,
                        energy_k,
                        pairing,
                    )
                    Delta_Iteration[n, 0] = (1 - mixing) * Delta_Iteration[n - 1, 0] + (
                        mixing * delta_sum
                    )  # mixing helps stop divergence

                    if (
                        np.absolute(Delta_Iteration[n, 0] - Delta_Iteration[n - 1, 0])
                    ) < tolerance:
                        Delta_Temperature[hopping, temp] = Delta_Iteration[n, 0]

                        if n == Delta_convergence:
                            raise ValueError("NOT CONVERGED AT [t, T] =", hopping, temp)
                        break
                Delta_Iteration[0, 0] = Delta_Temperature[hopping, temp]

        "Find Tc"
        for hopping in range(hopping_step):
            for temp in range(temperature_step):
                if (
                    Delta_Temperature[hopping, temp] / Delta_Temperature[hopping, 0]
                    < 0.1
                ):
                    Tc[hopping] = temperature[temp]
                    break

        return Delta_Temperature, Tc, temperature

    if gap_type == "two component":
        (
            Interaction_Potential_1,
            Interaction_Potential_2,
            pairing_x,
            pairing_y,
        ) = (
            gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge)[0],
            gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge)[1],
            gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge)[2],
            gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge)[3],
        )
        Iteration_Constant_xsq_ysq = -Interaction_Potential_1 / (
            2 * k_space_n
        )  # Combined constant for the integral sum
        Iteration_Constant_xy = -Interaction_Potential_2 / (
            2 * k_space_n
        )  # Combined constant for the integral sum

        "create arrays"

        Delta_Iteration_x = np.zeros((Delta_convergence, hopping_step))
        Delta_Iteration_y = np.zeros((Delta_convergence, hopping_step))
        Delta_Temperature_x = np.zeros((hopping_step, temperature_step))
        Delta_Temperature_y = np.zeros((hopping_step, temperature_step))
        Tc_x = np.zeros((hopping_step))
        Tc_y = np.zeros((hopping_step))
        Delta_pairing = np.zeros(
            (nk, nk), dtype=complex
        )  # Allow for complex order paramaters

        "Run calculation"
        for hopping in range(hopping_step):
            if hopping == 0:
                Delta_Iteration_x[0, 0], Delta_Iteration_y[0, 0] = (
                    Delta_guess,
                    Delta_guess,
                )
            else:
                # Tc and delta vary smoothly, thus this assumption saves convergence time
                Delta_Iteration_x[0, 0], Delta_Iteration_y[0, 0] = (
                    Delta_Temperature_x[hopping - 1, 0],
                    Delta_Temperature_y[hopping - 1, 0],
                )

            for temp in range(temperature_step):
                for n in range(1, Delta_convergence):
                    Delta_x = Delta_Iteration_x[n - 1, 0] * pairing_x[:, :]
                    Delta_y = 1j * Delta_Iteration_y[n - 1, 0] * pairing_y[:, :]
                    Delta_pairing[:, :] = Delta_x[:, :] + Delta_y[:, :]
                    Energy_k = np.sqrt(
                        epsilon_k_squared[hopping, :, :] + np.abs(Delta_pairing) ** 2
                    )
                    Fermi_dirac = 1 / (np.exp(Energy_k / boltzman_evt[temp]) + 1)

                    delta_sum_x = delta_sum_func(
                        Iteration_Constant_xsq_ysq,
                        np.real(Delta_pairing),
                        Fermi_dirac,
                        Energy_k,
                        pairing_x,
                    )
                    delta_sum_y = delta_sum_func(
                        Iteration_Constant_xy,
                        np.imag(Delta_pairing),
                        Fermi_dirac,
                        Energy_k,
                        pairing_y,
                    )
                    Delta_Iteration_x[n, 0] = (1 - mixing) * Delta_Iteration_x[
                        n - 1, 0
                    ] + (mixing * delta_sum_x)
                    Delta_Iteration_y[n, 0] = (1 - mixing) * Delta_Iteration_y[
                        n - 1, 0
                    ] + (mixing * delta_sum_y)
                    if (
                        np.absolute(
                            Delta_Iteration_x[n, 0] - Delta_Iteration_x[n - 1, 0]
                        )
                        + np.absolute(
                            Delta_Iteration_y[n, 0] - Delta_Iteration_y[n - 1, 0]
                        )
                    ) < tolerance:
                        Delta_Temperature_x[hopping, temp] = Delta_Iteration_x[n, 0]
                        Delta_Temperature_y[hopping, temp] = Delta_Iteration_y[n, 0]

                        if n == Delta_convergence:
                            raise ValueError("NOT CONVERGED AT [t, T] =", hopping, temp)
                        break
                Delta_Iteration_x[0, 0], Delta_Iteration_y[0, 0] = (
                    Delta_Temperature_x[hopping, temp],
                    Delta_Temperature_y[hopping, temp],
                )

        "Find Tc"
        for hopping in range(hopping_step):
            for temp in range(temperature_step):
                if (
                    Delta_Temperature_x[hopping, temp] / Delta_Temperature_x[hopping, 0]
                    < 0.1
                ):
                    Tc_x[hopping] = temperature[temp]
                    break

        for hopping in range(hopping_step):
            for temp in range(temperature_step):
                if (
                    Delta_Temperature_y[hopping, temp] / Delta_Temperature_y[hopping, 0]
                    < 0.1
                ):
                    Tc_y[hopping] = temperature[temp]
                    break

        return Delta_Temperature_x, Delta_Temperature_y, Tc_x, Tc_y, temperature


def dos(
    strain,
    nk,
    BZ_edge,
    dos_step,
    de,
    gap_pairing,
    a1x,
    a1y,
    gap_type,
    hopping_dual_range,
    hopping_step,
    e_k,
    epsilon,
    fermi_value,
):
    """
    :return: dos & FS weighted by the gap symmetry, hopping vals pertaining to specific DOS and FS curves presented
    """



    inv_k_space = 1 / (
        (nk**2) * np.pi
    )  # calculate inverse of k_space multiplied for pi to correctly convert this DOS int. to a DOS sum

    " Create zero arrays to be written into "
    "For loops to calculate the DOS for each hopping step, accross the enegry range, calling Delta as a func of temp"

    if gap_type == "one component":
        pairing = gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge)[1]
        gap_pairing_sq = pairing**2
        if hopping_dual_range:
            dos_NS_weighted = np.zeros((hopping_step))
            dos_nodal = np.zeros((3, nk, nk))

            for hopping in range(hopping_step):
                dos_NS_weighted[hopping] = inv_k_space * np.sum(
                    delta(de, 0 - e_k[hopping, :, :]) * gap_pairing_sq
                )

            dos_nodal[0, :, :] = (
                delta(de, 0 - e_k[fermi_value[0], :, :]) * gap_pairing_sq
            )  # Expanded Lattice
            dos_nodal[1, :, :] = (
                delta(de, 0 - e_k[fermi_value[1], :, :]) * gap_pairing_sq
            )  # zero strain lattice
            dos_nodal[2, :, :] = (
                delta(de, 0 - e_k[fermi_value[2], :, :]) * gap_pairing_sq
            )  # compressd lattice
        else:
            dos_NS_weighted = np.zeros((hopping_step, dos_step))
            dos_nodal = np.zeros((2, nk, nk))
            fermi_val = fermi_value[1]
            for hopping in range(hopping_step):
                for e_step in range(dos_step):
                    dos_NS_weighted[hopping, e_step] = inv_k_space * np.sum(
                        delta(de, epsilon[e_step] - e_k[hopping, :, :]) * gap_pairing_sq
                    )

            dos_nodal[0, :, :] = delta(de, 0 - e_k[0, :, :]) * gap_pairing_sq
            dos_nodal[1, :, :] = delta(de, 0 - e_k[fermi_val, :, :]) * gap_pairing_sq
        return dos_NS_weighted, dos_nodal, fermi_value

    if gap_type == "two component":
        gap_pairing_x, gap_pairing_y = (
            gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge)[2],
            gap_parameters(gap_pairing, strain, a1x, a1y, nk, BZ_edge)[3],
        )
        gap_pairing_x_sq, gap_pairing_y_sq = gap_pairing_x**2, gap_pairing_y**2

        if hopping_dual_range:
            "Create zero arrays to be written into"

            dos_NS_weighted_x = np.zeros((hopping_step))
            dos_NS_weighted_y = np.zeros_like(dos_NS_weighted_x)
            for hopping in range(hopping_step):
                dos_NS_weighted_x[hopping] = inv_k_space * np.sum(
                    delta(de, 0 - e_k[hopping, :, :]) * gap_pairing_x_sq
                )
                dos_NS_weighted_y[hopping] = inv_k_space * np.sum(
                    delta(de, 0 - e_k[hopping, :, :]) * gap_pairing_y_sq
                )

            dos_nodal_x = np.zeros((3, nk, nk))
            dos_nodal_y = np.zeros((3, nk, nk))
            dos_nodal_x[0, :, :] = (
                delta(de, 0 - e_k[fermi_value[0], :, :]) * gap_pairing_x_sq
            )
            dos_nodal_x[1, :, :] = (
                delta(de, 0 - e_k[fermi_value[1], :, :]) * gap_pairing_x_sq
            )
            dos_nodal_x[2, :, :] = (
                delta(de, 0 - e_k[fermi_value[2], :, :]) * gap_pairing_x_sq
            )
            dos_nodal_y[0, :, :] = (
                delta(de, 0 - e_k[fermi_value[0], :, :]) * gap_pairing_y_sq
            )
            dos_nodal_y[1, :, :] = (
                delta(de, 0 - e_k[fermi_value[1], :, :]) * gap_pairing_y_sq
            )
            dos_nodal_y[2, :, :] = (
                delta(de, 0 - e_k[fermi_value[2], :, :]) * gap_pairing_y_sq
            )

        else:
            dos_NS_weighted_x = np.zeros((hopping_step, dos_step))
            dos_NS_weighted_y = np.zeros_like(dos_NS_weighted_x)
            dos_starttime = datetime.now()
            "For loops to calculate the DOS for each hopping step, accross the enegry range, calling Delta as a func of temp"
            for hopping in range(hopping_step):
                for e_step in range(dos_step):
                    dos_NS_weighted_x[hopping, e_step] = inv_k_space * np.sum(
                        delta(de, epsilon[e_step] - e_k[hopping, :, :])
                        * gap_pairing_x_sq
                    )
                    dos_NS_weighted_y[hopping, e_step] = inv_k_space * np.sum(
                        delta(de, epsilon[e_step] - e_k[hopping, :, :])
                        * gap_pairing_y_sq
                    )

            dos_nodal_x = np.zeros((2, nk, nk))
            dos_nodal_y = np.zeros((2, nk, nk))
            fermi_val = fermi_value[1]
            dos_nodal_x[0, :, :] = delta(de, 0 - e_k[0, :, :]) * gap_pairing_x_sq
            dos_nodal_x[1, :, :] = (
                delta(de, 0 - e_k[fermi_val, :, :]) * gap_pairing_x_sq
            )
            dos_nodal_y[0, :, :] = delta(de, 0 - e_k[0, :, :]) * gap_pairing_y_sq
            dos_nodal_y[1, :, :] = (
                delta(de, 0 - e_k[fermi_val, :, :]) * gap_pairing_y_sq
            )

        return (
            dos_NS_weighted_x,
            dos_NS_weighted_y,
            dos_nodal_x,
            dos_nodal_y,
            fermi_value,
        )


def Brillouin_zone(strain, a1x, a1y, vxy):
    """
    inputs:
        - a1x,a1y: components of the lattice vectors, normalised i.e 0< a1i <1
    :return: BZ, 6x n grid that contains the lines perpendicular
    """

    if strain == "uniaxial":
        a2y = 1 - (vxy * (a1x - 1))
        a1 = np.array([a1x, a1y])
        a2 = np.array([a1y, a2y])

    if strain == "shear":
        a1 = np.array([a1x, a1y])
        a2 = np.array([a1y, a1x])

    if strain == "c_axis":
        a1 = np.array([a1x, a1y])
        a2 = np.array([a1y, a1x])

    b1 = (2.0 * np.pi / (a1[0] * a2[1] - a2[0] * a1[1])) * np.array([a2[1], -a2[0]])
    b2 = (2.0 * np.pi / (a1[0] * a2[1] - a2[0] * a1[1])) * np.array([-a1[1], a1[0]])

    # 2N x 2N lattice points
    N = 4
    nv = np.arange(-N, N)
    mv = np.arange(-N, N)
    # x-co-ordinates of the lattice points
    kxp = np.array([[i * b1[0] + j * b2[0] for i in nv] for j in mv])
    kyp = np.array([[i * b1[1] + j * b2[1] for i in nv] for j in mv])

    # all possible G vectors: (G * k = k ** 2 / 2)
    v1 = b1
    v2 = b2
    v3 = -b1
    v4 = -b2
    v5 = b2 - b1
    v6 = b1 - b2
    v7 = b1 + b2
    v8 = -v7

    # Vectors orthogonal to v1,v2,v3,v4,v5 and v6
    v1n = np.array([v1[1], -v1[0]])
    v2n = np.array([v2[1], -v2[0]])
    v3n = np.array([v3[1], -v3[0]])
    v4n = np.array([v4[1], -v4[0]])
    v5n = np.array([v5[1], -v5[0]])
    v6n = np.array([v6[1], -v6[0]])
    v7n = np.array([v7[1], -v7[0]])
    v8n = np.array([v8[1], -v8[0]])

    # Lines along v1,v2,v3,v4,v5 and v6
    line_length = 10

    l1 = [0.5 * v1 + i * v1n for i in np.linspace(-10.0, 10.0, line_length)]
    l2 = [0.5 * v2 + i * v2n for i in np.linspace(-10.0, 10.0, line_length)]
    l3 = [0.5 * v3 + i * v3n for i in np.linspace(-10.0, 10.0, line_length)]
    l4 = [0.5 * v4 + i * v4n for i in np.linspace(-10.0, 10.0, line_length)]
    l5 = [0.5 * v5 + i * v5n for i in np.linspace(-10.0, 10.0, line_length)]
    l6 = [0.5 * v6 + i * v6n for i in np.linspace(-10.0, 10.0, line_length)]
    l7 = [0.5 * v7 + i * v7n for i in np.linspace(-10.0, 10.0, line_length)]
    l8 = [0.5 * v8 + i * v8n for i in np.linspace(-10.0, 10.0, line_length)]

    return l1, l2, l3, l4, l5, l6, l7, l8, kxp, kyp


def main(
    strain, gap_pairing, t_1, t_2, hopping_step, energyband, sc_gap, calculate_dos
):
    """
    Calls all functions, kwargs must be strain, gap pairing, hopping range, Seperate plot function
    :return: Activate functions depending on kawrgs boolean values, saves outputs
    """
    # Set a few important exceptions raises
    if strain not in ["uniaxial", "shear", "c_axis"]:
        raise Exception(
            "Strain kwarg not valid",
            "\n Valid kwargs are: 'uniaxial', \n 'shear' \n or 'c_axis'",
        )
    if gap_pairing not in [
        "s_wave",
        "ext_s_wave",
        "d_wave",
        "d_plus_id",
        "d_plus_ig",
        "chiral_p",
    ]:
        raise Exception(
            "gap_pairing kwarg not valid",
            "'s_wave', \n 'ext_s_wave', \n 'd_wave', \n 'd_plus_id', \n 'd_plus_ig', \n'chiral_p'",
        )
    if not os.path.isfile(f"Energyband_{strain}.npy") and energyband == False:
        raise FileNotFoundError("Energyband file not found AND not calculated")



    full_title, gap_type, x_title, y_title = gap_titles(gap_pairing)

    # Define code parameters
    n_k = 700  # k-space grid size s.t. n_x, n_y = n_k, n_k
    temperature_step = 150  # number of steps in temperature range
    tolerance = (
        1.8 * 10**-8
    )  # Set tolerance for testing convergence, smaller values give greater accuracy
    mixing = 0.8  # mixing for self conistent caculation, higher mixing keeps the value closer to the guess
    Delta_convergence = 2000  # Number of iterations on the self consistent loop
    dos_step = 2000  # number of incriments on the energy range in the dos calculation
    de = 1 * 10**-3  # screening energy to calculate density of states approximaiton

    # Define phsycial parameter values
    e_0 = -131 * 10**-3  # initial onsite//fermi energy value
    t_x = +81.62 * 10**-3  # Hopping integral value xdir, nearest neighbour
    t_prime = +36.73 * 10**-3  # Second-nearest neighbour hopping integral
    v_xy = -0.42  # Poission ratio used for relating hopping integrals
    BZ_edge = 1
    a1x = 1  # x component of lattice vector a1
    a1y = 0  # y component of lattice vector a1

    Delta_guess = 0.2 * 10**-3  # Initial Guess for Delta to be iterated over
    temperature_max = 3  # Maximum temperature in kelvin
    epsilon = np.linspace(
        -2 * t_x, 2.5 * t_x, dos_step
    )  # energy range of width of DOS to calculate
    fermi_value = np.zeros((3), dtype=int)
    fermi_value[0], fermi_value[1], fermi_value[2] = 0, 2, 3

    # Load k-space grid for all calculations
    kx, ky, k_x, k_y = k_space(n_k, BZ_edge)

    if energyband:
        e_k_start_time = datetime.now()
        e_k, e_k_fig, hopping_array, hopping_parameter = epsilon_k(
            strain,
            n_k,
            t_1,
            t_2,
            hopping_step,
            t_x,
            t_prime,
            e_0,
            BZ_edge,
            kx,
            ky,
            k_x,
            k_y,
            v_xy,
            plot_nth_contour=-1,
        )

        "Save energyband"
        np.save("Energyband_{}".format(strain), epsilon_k)
        if strain == "shear":
            np.save("hopping_array_{}".format(strain), hopping_array)
        else:
            np.save("hopping_array_{}".format(strain), hopping_array)

        l1, l2, l3, l4, l5, l6, l7, l8, kxp, kyp = Brillouin_zone(
            strain, a1x, a1y, v_xy
        )
        e_k_fig.plot([i[0] for i in l1], [i[1] for i in l1], "k--")
        e_k_fig.plot([i[0] for i in l2], [i[1] for i in l2], "k--")
        e_k_fig.plot([i[0] for i in l3], [i[1] for i in l3], "k--")
        e_k_fig.plot([i[0] for i in l4], [i[1] for i in l4], "k--")
        e_k_fig.plot([i[0] for i in l5], [i[1] for i in l5], "k--")
        e_k_fig.plot([i[0] for i in l6], [i[1] for i in l6], "k--")
        e_k_fig.plot([i[0] for i in l7], [i[1] for i in l7], "k--")
        e_k_fig.plot([i[0] for i in l8], [i[1] for i in l8], "k--")
        e_k_fig.text(0.0, 0.7, "$\Gamma$", fontsize=22)
        e_k_fig.set_xlim(-1.2 * BZ_edge * np.pi, 1.2 * BZ_edge * np.pi)
        e_k_fig.set_ylim(-1.2 * BZ_edge * np.pi, 1.2 * BZ_edge * np.pi)

        plt.savefig("energyband.jpg")

        e_k_end_time = datetime.now()
        print(f"\nEnergy band runtime = {e_k_end_time - e_k_start_time}")

    else:
        e_k = np.load(f"Energyband_{strain}.npy")

    hopping_array, symmetric_hopping, t_unstrained = get_hopping_range(
        strain, t_x, t_prime
    )

    n_1 = 2
    n_2 = 4
    t_x_1 = round(
        hopping_array[n_1] / t_unstrained, 2
    )  # Converts code number into a strain
    t_x_2 = round(hopping_array[n_2] / t_unstrained, 2)
    t_x_end = round(hopping_array[-1] / t_unstrained, 2)

    if gap_type == "one component":
        print("one comp init")

        if calculate_dos:
            dos_start_time = datetime.now()
            dos_weighted, fermi_surface, fermi_value = dos(
                strain,
                n_k,
                BZ_edge,
                dos_step,
                de,
                gap_pairing,
                a1x,
                a1y,
                gap_type,
                symmetric_hopping,
                hopping_step,
                e_k,
                epsilon,
                fermi_value,
            )
            dos_end_time = datetime.now()
            print(f"\n DOS runtime = {dos_end_time - dos_start_time}")

            if symmetric_hopping:
                hopping_1, hopping_2, hopping_3 = round_fermi_vals(
                    strain, hopping_array, fermi_value
                )

                "create figure"
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
                ax1.contour(k_x, k_y, fermi_surface[1, :, :], colors="blue")
                cntr1 = ax1.contour(k_x, k_y, fermi_surface[0, :, :], colors="red")
                ax1.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax1.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])
                ax2.contour(k_x, k_y, fermi_surface[1, :, :], colors="blue")
                cntr2 = ax2.contour(k_x, k_y, fermi_surface[-1, :, :], colors="red")
                ax2.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax2.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])
                h1, _ = cntr1.legend_elements()
                h2, _ = cntr2.legend_elements()
                ax1.legend(
                    [h1[0]],
                    ["$t_x/t_0$ = {}".format(hopping_1)],
                    loc="center",
                    fontsize=16,
                )
                ax2.legend(
                    [h2[0]],
                    ["$t_x/t_0$ = {}".format(hopping_3)],
                    loc="center",
                    fontsize=16,
                )
                ax1.xaxis.set_tick_params(
                    labelsize=25, labeltop=False, labelbottom=True
                )
                ax1.yaxis.set_tick_params(
                    labelsize=25, labelright=False, labelleft=True
                )
                ax2.xaxis.set_tick_params(
                    labelsize=25, labeltop=True, labelbottom=False
                )
                ax2.yaxis.set_tick_params(
                    labelsize=25, labelright=True, labelleft=False
                )
                ax1.set_xlabel("k$_x$", fontsize=25)
                ax1.set_ylabel("k$_y$", fontsize=25)
                ax1.set_title(f"{full_title}", fontsize=32)
                plt.savefig(f"{strain}_dos_{gap_pairing}.jpg")
            else:
                fig, ax1 = plt.subplots(figsize=(8, 8))
                left, bottom, width, height = [0.15, 0.61, 0.225, 0.225]  # top left
                # left, bottom, width, height = [0.62, 0.25, 0.225, 0.225] #bottom right
                ax2 = fig.add_axes([left, bottom, width, height])

                "Plot ax1, dos vs energy at various strains"
                ax1.plot(
                    epsilon[:],
                    dos_weighted[0, :],
                    "b.",
                    label=(hopping_array[0] / hopping_array[0]),
                )
                ax1.plot(epsilon[:], dos_weighted[n_1, :], "g.", label=(t_x_1))
                ax1.plot(epsilon[:], dos_weighted[n_2, :], "r.", label=(t_x_2))
                ax1.plot(epsilon[:], dos_weighted[-1, :], "m.", label=(t_x_end))
                ax1.set_xticks((-t_x, 0, t_x), ["$-t_0$", "$\epsilon_{f}$", "$t_0$"])
                ax1.xaxis.set_tick_params(labelsize=28)
                ax1.set_xlabel("$\\varepsilon$ ", fontsize=32)
                ax1.set_ylabel("$\\rho(\\varepsilon)$", fontsize=36)

                ax2.contour(k_x, k_y, fermi_surface[1, :, :], colors="red")
                ax2.xaxis.set_tick_params(
                    labelsize=15, labeltop=True, labelbottom=False
                )
                ax2.yaxis.set_tick_params(
                    labelsize=15, labelright=True, labelleft=False
                )
                ax2.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax2.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])

                l1, l2, l3, l4, l5, l6, l7, l8, kxp, kyp = Brillouin_zone(
                    strain, a1x, a1y, v_xy
                )
                ax2.plot([i[0] for i in l1], [i[1] for i in l1], "k--")
                ax2.plot([i[0] for i in l2], [i[1] for i in l2], "k--")
                ax2.plot([i[0] for i in l3], [i[1] for i in l3], "k--")
                ax2.plot([i[0] for i in l4], [i[1] for i in l4], "k--")
                ax2.plot([i[0] for i in l5], [i[1] for i in l5], "k--")
                ax2.plot([i[0] for i in l6], [i[1] for i in l6], "k--")
                ax2.plot([i[0] for i in l7], [i[1] for i in l7], "k--")
                ax2.plot([i[0] for i in l8], [i[1] for i in l8], "k--")
                # ax2.text(0.0, .7, "$\Gamma$", fontsize=22)
                ax2.set_xlim(-1.2 * np.pi, 1.2 * np.pi)
                ax2.set_ylim(-1.2 * np.pi, 1.2 * np.pi)
                ax2.contour(k_x, k_y, fermi_surface[0, :, :], colors="blue")
                ax1.set_title(f"{full_title}", fontsize=32)
                plt.savefig(f"{strain}_dos_{gap_pairing}.jpg")

            np.save("{}_{}_fermi_surface".format(gap_pairing, strain), fermi_surface)

        if sc_gap:
            delta_start_time = datetime.now()
            delta, tc, temperature = delta_calculation(
                strain,
                gap_type,
                e_k,
                n_k,
                Delta_convergence,
                hopping_step,
                temperature_max,
                temperature_step,
                n_k,
                kx,
                ky,
                tolerance,
                mixing,
                Delta_guess,
                gap_pairing,
                a1x,
                a1y,
                BZ_edge,
            )
            np.save("{}_tc_{}".format(gap_pairing, strain), tc)
            np.save("{}_delta_{}".format(gap_pairing, strain), delta)
            np.save("{}_Temp_{}".format(gap_pairing, strain), temperature)
            delta_end_time = datetime.now()
            print(f"\n$\Delta$ runtime = {delta_end_time - delta_start_time}")

            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(hopping_array[:] / t_unstrained, tc[:], "r.", markersize=16)
            ax1.set_ylim((0, np.max(tc) + 0.1))
            ax1.locator_params(axis="x", nbins=8)
            ax1.xaxis.set_tick_params(labelsize=15)
            ax1.yaxis.set_tick_params(labelsize=15)
            ax1.set_ylabel("Tc (k)", fontsize=28)
            if strain == "shear":
                ax1.set_xlabel("$t_{xy}^{(1)} / t^{'}$", fontsize=28)
            else:
                ax1.set_xlabel("$t_x / t_0$", fontsize=32)
            ax1.locator_params(axis="y", nbins=8)
            ax1.set_title(f"{full_title}", fontsize=32)
            if symmetric_hopping:
                plt.savefig(f"{strain}_tc_{gap_pairing}_full.jpg")
            else:
                plt.savefig(f"{strain}_delta_{gap_pairing}.jpg")

    if gap_type == "two component":
        print("two comp init")

        if calculate_dos:
            dos_start_time = datetime.now()
            x_dos, y_dos, fermi_x, fermi_y, hopping_vals = dos(
                strain,
                n_k,
                BZ_edge,
                dos_step,
                de,
                gap_pairing,
                a1x,
                a1y,
                gap_type,
                symmetric_hopping,
                hopping_step,
                e_k,
                epsilon,
                fermi_value,
            )
            dos_end_time = datetime.now()
            print(f"\n DOS runtime = {dos_end_time - dos_start_time}")
            if symmetric_hopping:
                np.save("{}_x_dos_e_fermi_{}".format(gap_pairing, strain), x_dos)
                np.save("{}_y_dos_e_fermi_{}".format(gap_pairing, strain), y_dos)
            else:
                np.save("{}_x_dos_{}".format(gap_pairing, strain), x_dos)
                np.save("{}_y_dos_{}".format(gap_pairing, strain), y_dos)
            np.save("{}_x_fermi_surface_{}".format(gap_pairing, strain), fermi_x)
            np.save("{}_y_fermi_surface_{}".format(gap_pairing, strain), fermi_y)
            np.save(
                "{}_{}_fermi_hopping_value".format(gap_pairing, strain), hopping_vals
            )
            np.save(
                "{}_{}_fermi_hopping_value".format(gap_pairing, strain), fermi_value
            )

            if symmetric_hopping:
                "hopping vals to match strain displayed"
                hopping_1, hopping_2, hopping_3 = round_fermi_vals(
                    strain,
                    hopping_array,
                    np.load(
                        "{}_{}_fermi_hopping_value.npy".format(gap_pairing, strain)
                    ),
                )

                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

                ax1.contour(k_x, k_y, fermi_x[1, :, :], colors="blue")
                cntr1 = ax1.contour(k_x, k_y, fermi_x[0, :, :], colors="red")
                ax1.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax1.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])

                ax2.contour(k_x, k_y, fermi_x[1, :, :], colors="blue")
                cntr2 = ax2.contour(k_x, k_y, fermi_x[-1, :, :], colors="red")
                ax2.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax2.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])
                h1, _ = cntr1.legend_elements()
                h2, _ = cntr2.legend_elements()
                ax1.legend(
                    [h1[0]],
                    ["$t_x/t_0$ = {}".format(hopping_1)],
                    loc="center",
                    fontsize=16,
                )
                ax2.legend(
                    [h2[0]],
                    ["$t_x/t_0$ = {}".format(hopping_3)],
                    loc="center",
                    fontsize=16,
                )
                ax1.xaxis.set_tick_params(
                    labelsize=25, labeltop=False, labelbottom=True
                )
                ax1.yaxis.set_tick_params(
                    labelsize=25, labelright=False, labelleft=True
                )
                ax2.xaxis.set_tick_params(
                    labelsize=25, labeltop=True, labelbottom=False
                )
                ax2.yaxis.set_tick_params(
                    labelsize=25, labelright=True, labelleft=False
                )
                ax1.set_xlabel("k$_x$", fontsize=25)
                ax1.set_ylabel("k$_y$", fontsize=25)
                plt.savefig(f"{strain}_{gap_pairing}_x_dos_weighted.jpg")
                plt.clf()

                fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 8))
                ax3.contour(k_x, k_y, fermi_y[1, :, :], colors="blue")
                cntr3 = ax3.contour(k_x, k_y, fermi_y[0, :, :], colors="red")
                ax3.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax3.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])

                ax4.contour(k_x, k_y, fermi_y[1, :, :], colors="blue")
                cntr4 = ax4.contour(k_x, k_y, fermi_y[-1, :, :], colors="red")
                ax4.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax4.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])
                h3, _ = cntr3.legend_elements()
                h4, _ = cntr4.legend_elements()
                ax3.legend(
                    [h3[0]],
                    ["$t_x/t_0$ = {}".format(hopping_1)],
                    loc="center",
                    fontsize=16,
                )
                ax4.legend(
                    [h4[0]],
                    ["$t_x/t_0$ = {}".format(hopping_3)],
                    loc="center",
                    fontsize=16,
                )
                ax3.xaxis.set_tick_params(
                    labelsize=25, labeltop=False, labelbottom=True
                )
                ax3.yaxis.set_tick_params(
                    labelsize=25, labelright=False, labelleft=True
                )
                ax4.xaxis.set_tick_params(
                    labelsize=25, labeltop=True, labelbottom=False
                )
                ax4.yaxis.set_tick_params(
                    labelsize=25, labelright=True, labelleft=False
                )
                ax3.set_xlabel("k$_x$", fontsize=25)
                ax3.set_ylabel("k$_y$", fontsize=25)
                plt.savefig(f"{strain}_{gap_pairing}_y_dos_weighted.jpg")
                plt.clf()

            else:

                " Create figure and axis sizes "
                fig1, ax1 = plt.subplots(figsize=(8, 8))


                left, bottom, width, height = [0.62, 0.61, 0.225, 0.225]
                ax2 = fig1.add_axes([left, bottom, width, height])

                "Plot ax1, dos vs energy at various strains"
                ax1.plot(
                    epsilon[:],
                    x_dos[0, :],
                    "b.",
                    label=(hopping_array[0] / hopping_array[0]),
                )
                ax1.plot(epsilon[:], x_dos[n_1, :], "g.", label=(t_x_1))
                ax1.plot(epsilon[:], x_dos[n_2, :], "r.", label=(t_x_2))
                ax1.plot(epsilon[:], x_dos[-1, :], "m.", label=(t_x_end))
                ax1.set_xticks((-t_x, 0, t_x), ["$-t_0$", "$\epsilon_{f}$", "$t_0$"])
                ax1.xaxis.set_tick_params(labelsize=28)
                ax1.set_xlabel("$\\varepsilon$", fontsize=32)
                ax1.set_ylabel("$\\rho(\\varepsilon)$", fontsize=36)
                if strain == "shear":
                    ax1.legend(
                        loc="upper left",
                        title="$\\frac{t_{xy^{(1)}}}{t^{'}}$",
                        prop={"size": 20},
                        title_fontsize=28,
                        markerscale=4,
                    )
                else:
                    ax1.legend(
                        loc="upper right",
                        title="$\\frac{t_x}{t_0}$",
                        prop={"size": 20},
                        title_fontsize=28,
                        markerscale=4,
                    )

                "ax2 plot, Fermi-surface inset"

                ax2.contour(k_x, k_y, fermi_x[1, :, :], colors="red")

                ax2.xaxis.set_tick_params(
                    labelsize=15, labeltop=True, labelbottom=False
                )
                ax2.yaxis.set_tick_params(
                    labelsize=15, labelright=True, labelleft=False
                )
                ax2.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax2.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])

                l1, l2, l3, l4, l5, l6, l7, l8, kxp, kyp = Brillouin_zone(
                    strain, a1x, a1y, v_xy
                )
                ax2.plot([i[0] for i in l1], [i[1] for i in l1], "k--")
                ax2.plot([i[0] for i in l2], [i[1] for i in l2], "k--")
                ax2.plot([i[0] for i in l3], [i[1] for i in l3], "k--")
                ax2.plot([i[0] for i in l4], [i[1] for i in l4], "k--")
                ax2.plot([i[0] for i in l5], [i[1] for i in l5], "k--")
                ax2.plot([i[0] for i in l6], [i[1] for i in l6], "k--")
                ax2.plot([i[0] for i in l7], [i[1] for i in l7], "k--")
                ax2.plot([i[0] for i in l8], [i[1] for i in l8], "k--")
                # ax2.text(0.0, .7, "$\Gamma$", fontsize=22)
                ax2.set_xlim(-1.2 * np.pi, 1.2 * np.pi)
                ax2.set_ylim(-1.2 * np.pi, 1.2 * np.pi)

                ax2.contour(k_x, k_y, fermi_x[0, :, :], colors="blue")
                plt.savefig(f"{strain}_{gap_pairing}_x_dos_weighted.jpg")

                " Create figure and axis sizes "
                fig2, ax3 = plt.subplots(figsize=(8, 8))
                # These are in unitless percentages of the figure size. (0,0 is bottom left)
                left, bottom, width, height = [0.62, 0.23, 0.225, 0.225]  # bottom right

                ax4 = fig2.add_axes([left, bottom, width, height])

                "Plot ax1, dos vs energy at various strains"
                ax3.plot(
                    epsilon[:],
                    y_dos[0, :],
                    "b.",
                    label=(hopping_array[0] / hopping_array[0]),
                )
                ax3.plot(epsilon[:], y_dos[n_1, :], "g.", label=(t_x_1))
                ax3.plot(epsilon[:], y_dos[n_2, :], "r.", label=(t_x_2))
                ax3.plot(epsilon[:], y_dos[-1, :], "m.", label=(t_x_end))
                ax3.set_xticks((-t_x, 0, t_x), ["$-t_0$", "$\epsilon_{f}$", "$t_0$"])
                ax3.xaxis.set_tick_params(labelsize=28)
                ax3.set_xlabel("$\\varepsilon$", fontsize=32)
                ax3.set_ylabel("$\\rho(\\varepsilon)$", fontsize=36)
                if strain == "shear":
                    ax3.legend(
                        loc="upper right",
                        title="$\\frac{t_{xy^{(1)}}}{t^{'}}$",
                        prop={"size": 20},
                        title_fontsize=28,
                        markerscale=4,
                    )
                else:
                    ax3.legend(
                        loc="upper right",
                        title="$\\frac{t_x}{t_0}$",
                        prop={"size": 20},
                        title_fontsize=28,
                        markerscale=4,
                    )

                "ax2 plot, Fermi-surface inset"

                ax4.contour(k_x, k_y, fermi_y[1, :, :], colors="red")
                ax4.xaxis.set_tick_params(
                    labelsize=15, labeltop=True, labelbottom=False
                )
                ax4.yaxis.set_tick_params(
                    labelsize=15, labelright=True, labelleft=False
                )
                ax4.set_xticks((-np.pi, 0, np.pi), ["$-\pi$", "0", "$\pi$"])
                ax4.set_yticks((np.pi, 0, -np.pi), ["$\pi$", "0", "$-\pi$"])

                l1, l2, l3, l4, l5, l6, l7, l8, kxp, kyp = Brillouin_zone(
                    strain, a1x, a1y, v_xy
                )
                ax4.plot([i[0] for i in l1], [i[1] for i in l1], "k--")
                ax4.plot([i[0] for i in l2], [i[1] for i in l2], "k--")
                ax4.plot([i[0] for i in l3], [i[1] for i in l3], "k--")
                ax4.plot([i[0] for i in l4], [i[1] for i in l4], "k--")
                ax4.plot([i[0] for i in l5], [i[1] for i in l5], "k--")
                ax4.plot([i[0] for i in l6], [i[1] for i in l6], "k--")
                ax4.plot([i[0] for i in l7], [i[1] for i in l7], "k--")
                ax4.plot([i[0] for i in l8], [i[1] for i in l8], "k--")
                ax4.set_xlim(-1.2 * np.pi, 1.2 * np.pi)
                ax4.set_ylim(-1.2 * np.pi, 1.2 * np.pi)

                ax4.contour(k_x, k_y, fermi_y[0, :, :], colors="blue")
                plt.savefig(f"{strain}_{gap_pairing}_y_dos_weighted.jpg")

        if sc_gap:
            delta_start_time = datetime.now()
            delta_x, delta_y, Tc_x, Tc_y, temperature = delta_calculation(
                strain,
                gap_type,
                e_k,
                n_k,
                Delta_convergence,
                hopping_step,
                temperature_max,
                temperature_step,
                n_k,
                kx,
                ky,
                tolerance,
                mixing,
                Delta_guess,
                gap_pairing,
                a1x,
                a1y,
                BZ_edge,
            )

            np.save("{}_tc_x_{}".format(gap_pairing, strain), Tc_x)
            np.save("{}_tc_y_{}".format(gap_pairing, strain), Tc_y)
            np.save("{}_delta_x_{}".format(gap_pairing, strain), delta_x)
            np.save("{}_delta_y_{}".format(gap_pairing, strain), delta_y)
            np.save("{}_Temp_{}".format(gap_pairing, strain), temperature)
            delta_end_time = datetime.now()
            print(f"\n $\Delta$ runtime = {delta_end_time - delta_start_time}")

            tc_x_max, tc_y_max = np.max(Tc_x), np.max(Tc_y)
            tc_max = np.maximum(tc_x_max, tc_y_max)

            if symmetric_hopping:
                if not os.path.isfile(
                    f"{gap_pairing}_x_dos_e_fermi_{strain}.npy"
                ) and not os.path.isfile(f"{gap_pairing}_y_dos_e_fermi_{strain}.npy"):
                    raise FileNotFoundError(
                        "Required DOS for plot inset not found, Set calculate_dos=True and run again"
                    )
                dos_x = np.load("{}_x_dos_e_fermi_{}.npy".format(gap_pairing, strain))
                dos_y = np.load("{}_y_dos_e_fermi_{}.npy".format(gap_pairing, strain))
                fig, ax1 = plt.subplots(figsize=(8, 8))
                ax1.plot(
                    hopping_array[:] / t_unstrained,
                    Tc_x[:],
                    "r.",
                    markersize=16,
                    label=x_title,
                )
                ax1.plot(
                    hopping_array[:] / t_unstrained,
                    Tc_y[:],
                    "b.",
                    markersize=16,
                    label=y_title,
                )
                ax1.xaxis.set_tick_params(labelsize=15)
                ax1.set_ylim((0, tc_max + 0.1))
                ax1.set_xticks((0.88, 1, 1.08, 1.24))
                ax1.yaxis.set_tick_params(labelsize=19)
                ax1.set_ylabel("Tc (k)", fontsize=32)
                if strain == "shear":
                    ax1.set_xlabel("$t_{xy}^{(1)} / t^{'}$", fontsize=28)
                else:
                    ax1.set_xlabel("$t_x / t_0$", fontsize=32)

                left_1, bottom_1, width_1, height_1 = [0.7, 0.655, 0.2, 0.225]
                ax2 = fig.add_axes([left_1, bottom_1, width_1, height_1])
                ax2.set_yticks([])
                ax2.set_xticks([])
                ax2.set_xticks([])
                ax2.plot(hopping_array[:] / t_unstrained, dos_y[:], "b.")


                ax2.set_ylabel("$\\rho(\\varepsilon_{f})$", fontsize=15)
                ax1.legend(loc="right", bbox_to_anchor=(1.02, 0.55), fontsize="20")
                ax1.legend(loc="right", bbox_to_anchor=(1.02, 0.09), fontsize="20")
                plt.savefig(f"{strain}_tc_{gap_pairing}_full.jpg")

            else:
                fig, ax1 = plt.subplots(figsize=(8, 8))
                ax1.plot(
                    hopping_array[:] / t_unstrained,
                    Tc_x[:],
                    "r.",
                    markersize=16,
                    label=x_title,
                )
                ax1.plot(
                    hopping_array[:] / t_unstrained,
                    Tc_y[:],
                    "b.",
                    markersize=16,
                    label=y_title,
                )
                ax1.set_ylim((0, tc_max + 0.1))
                ax1.xaxis.set_tick_params(labelsize=15)
                ax1.locator_params(axis="x", nbins=8)
                ax1.yaxis.set_tick_params(labelsize=19)
                ax1.set_ylabel("Tc (k)", fontsize=32)
                if strain == "shear":
                    ax1.set_xlabel("$t_{xy}^{(1)} / t^{'}$", fontsize=28)
                else:
                    ax1.set_xlabel("$t_x / t_0$", fontsize=32)
                ax1.legend(fontsize="30")
                plt.savefig(f"{strain}_tc_{gap_pairing}.jpg")


main(
    strain="shear",
    gap_pairing="s_wave",
    t_1=1,
    t_2=1.14,
    hopping_step=5,
    energyband=True,
    sc_gap=True,
    calculate_dos=True,
)

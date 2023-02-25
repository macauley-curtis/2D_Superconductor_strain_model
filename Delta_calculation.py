import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from math import log10, floor

" Create functions "

def delta_sum(C, Delta, fe, Ek, pairing):
    "Returns the gap standard equation"
    return C * np.sum(Delta * pairing * (1 - 2 * fe) / Ek)


def gap_symmetry(gap_pairing, kx, ky):
    "Returns the interaction potential(s) and the gap pairing(s)"
    if gap_pairing == "s_wave":
        return -0.0335, 1
    if gap_pairing == "d_wave":
        return -0.0394 * 0.5, (np.cos(kx[:, :]) - np.cos(ky[:, :]))
    if gap_pairing == "ext_s_wave":
        return -0.06483, np.cos(kx[:, :]) + np.cos(ky[:, :])
    if gap_pairing == "d_plus_id":
        return - 0.5 * 0.03915, - 0.5 * 0.323549,  np.cos(kx[:, :]) - np.cos(ky[:, :]),  np.sin(kx[:, :]) * np.sin(ky[:, :])
    if gap_pairing == "d_plus_ig":
        return -0.5 * 0.03928, -0.5 * 0.496063, (np.cos(kx[:, :]) - np.cos(ky[:, :])), (np.sin(kx) * np.sin(ky)) * (np.cos(kx[:, :]) - np.cos(ky[:, :]))
    if gap_pairing == "chiral_p":
        return -0.07501, -0.07501, np.sin(kx[:, :]), np.sin(ky[:, :])


def delta(strain, gap_pairing, temperature_step, temperature_max, save, tc_plot):
    "saves gap, tc and temperature for chosen strain and gap pairing"

    " Load input array"
    epsilon_k = np.load("Energyband_{}.npy".format(strain))
    hopping_step, n_x, n_y = np.shape(epsilon_k) #sets array dimensions to carry forward

    "define the temperature range"
    temperature = np.linspace(0, temperature_max, temperature_step)

    "Define how many components to model"
    if gap_pairing == "s_wave" or gap_pairing == "ext_s_wave" or gap_pairing == "d_wave":
        gap_type = "one component"
    else:
        gap_type = "two component"

    "Initiate main"
    if gap_type == "one component":
        delta, Tc, runtime = one_comp(save, gap_pairing, gap_type, tc_plot, n_x, n_y, hopping_step, temperature, temperature_step, epsilon_k,
                                      Delta_convergence=2000)
    if gap_type == "two component":
        delta_x, delta_y, Tc_x, Tc_y, runtime = two_comp(save, gap_pairing, gap_type, tc_plot, n_x, n_y, hopping_step,
                                                     temperature, temperature_step, epsilon_k, Delta_convergence=2000)
    "Save data"
    if save and gap_type == "one component":
        np.save("{}_tc_{}".format(gap_pairing, strain), Tc)
        np.save("{}_delta_{}".format(gap_pairing, strain), delta)
        np.save("{}_Temp_{}".format(gap_pairing, strain), temperature)

    if save and gap_type == "two component":
        np.save("{}_tc_x_{}".format(gap_pairing, strain), Tc_x)
        np.save("{}_tc_y_{}".format(gap_pairing, strain), Tc_y)
        np.save("{}_delta_x_{}".format(gap_pairing, strain), delta_x)
        np.save("{}_delta_y_{}".format(gap_pairing, strain), delta_y)
        np.save("{}_Temp_{}".format(gap_pairing, strain), temperature)

    "plot data"
    if tc_plot and gap_type == "one component":
        plt.plot(temperature, delta[0, :] * 10 ** 3, 'r.')
        plt.ylabel("$\Delta$ (meV)", fontsize=20)
        plt.xlabel("T (k)", fontsize=20)
        plt.title("uniaxial {}".format(gap_pairing))

    if tc_plot and gap_type == "two component":
        plt.plot(temperature, delta_x[0, :] * 10 ** 3, 'b.', label="$\Delta_x$")
        plt.plot(temperature, delta_y[0, :] * 10 ** 3, 'r.', label="$\Delta_y$")
        plt.legend()
        plt.ylabel("$\Delta$ (meV)", fontsize=20)
        plt.xlabel("T (k)", fontsize=20)
        plt.title("uniaxial {}".format(gap_pairing))
    print("runtime =", runtime)
    plt.show()


"One component calculations"
def one_comp(save, gap_pairing, gap_type, tc_plot, n_x, n_y, hopping_step, temperature, temperature_step, epsilon_k,
                                      Delta_convergence=2000):
    "Returns delta, tc and temperature"

    "If the gap type is incompatible this kills the function being called before errors are raise"
    if gap_type == "two component":
        exit()

    startTime = datetime.now()

    " Set code parameters for consistency loop "
    k_space_n = n_x * n_y  # Size of k space
    tolerance = 1.8 * 10 ** -9  # Set tolerance for testing convergence, smaller values give greater accuracy
    mixing = 0.8  # mixing for self conistent caculation, higher mixing keeps the value closer to the guess
    Delta_guess = 0.2 * 10 ** -3  # Initial Guess for Delta to be iterated over

    "create zero arrays"
    Delta_Iteration = np.zeros((Delta_convergence, hopping_step))
    epsilon_k_squared = epsilon_k ** 2
    Delta_Temperature = np.zeros((hopping_step, temperature_step))
    Tc = np.zeros((hopping_step))

    " Create k-space to be opperated on array-like"
    k_x = np.linspace(-np.pi, np.pi, n_x)
    k_y = np.linspace(+np.pi, -np.pi, n_y)
    kx = np.zeros((n_x, n_y))
    ky = np.zeros((np.shape(kx)))

    for i in range(n_x):
        kx[i, :] = k_x[:]
        ky[:, i] = k_y[:]

    " Set Physical Parameters"
    Interaction_Potential, pairing = gap_symmetry(gap_pairing, kx, ky)  #Import gap speicifc potential and symmetry
    Iteration_Constant = (-Interaction_Potential / (2 * k_space_n))  # Combined constant for the integral sum
    boltzman_evt = (1.38 * 10 ** -23) * temperature / (1.6 * 10 ** -19)  # Convert into Boltzman eV units for ease

    "Run calculation"
    for hopping in range(hopping_step):
        if hopping == 0:
            Delta_Iteration[0, 0] = Delta_guess
        else:
            Delta_Iteration[0, 0] = Delta_Temperature[hopping - 1, 0] #Tc and delta vary smoothly, thus this assumption saves convergence time

        for temp in range(temperature_step):

            for n in range(1, Delta_convergence):

                Delta_pairing = Delta_Iteration[n - 1, 0] * pairing
                Energy_k = np.sqrt(epsilon_k_squared[hopping, :, :] + np.abs(Delta_pairing) ** 2)
                Fermi_dirac = 1 / (np.exp(Energy_k / boltzman_evt[temp]) + 1)

                Delta_Sum_x = delta_sum(Iteration_Constant, Delta_pairing, Fermi_dirac, Energy_k, pairing)
                Delta_Iteration[n, 0] = (1 - mixing) * Delta_Iteration[n - 1, 0] + (mixing * Delta_Sum_x) #mixing helps stop divergence

                if (np.absolute(Delta_Iteration[n, 0] - Delta_Iteration[n - 1, 0])) < tolerance:
                    Delta_Temperature[hopping, temp] = Delta_Iteration[n, 0]

                    if n == Delta_convergence:
                        print("NOT CONVERGED AT [t, T] =", hopping, temp)
                    break
            Delta_Iteration[0, 0] = Delta_Temperature[hopping, temp]

    "Find Tc"
    for hopping in range(hopping_step):
        for temp in range(temperature_step):
            if Delta_Temperature[hopping, temp] / Delta_Temperature[hopping, 0] < 0.05:
                Tc[hopping] = temperature[temp]
                break

    return Delta_Temperature, Tc, datetime.now() - startTime

"two component calculations"
def two_comp(save, gap_pairing, gap_type, tc_plot, n_x, n_y, hopping_step,
                                                             temperature, temperature_step, epsilon_k, Delta_convergence=2000):
    "Returns delta, tc and temperature"

    "If the gap type is incompatible this kills the function being called before errors are raise"
    if gap_type == "one component":
        exit()
    startTime = datetime.now()

    " Set code parameters "
    k_space_n = n_x * n_y  # Size of k space
    tolerance = 1.8 * 10 ** -9  # Set tolerance for testing convergence, smaller values give greater accuracy
    mixing = 0.8  # mixing for self conistent caculation, higher mixing keeps the value closer to the guess
    Delta_guess = 0.2 * 10 ** -3  # Initial Guess for Delta to be iterated over

    " Create k-space to be opperated on array-like"
    k_x = np.linspace(-np.pi, np.pi, n_x)
    k_y = np.linspace(+np.pi, -np.pi, n_y)
    kx = np.zeros((n_x, n_y))
    ky = np.zeros((np.shape(kx)))

    for i in range(n_x):
        kx[i, :] = k_x[:]
        ky[:, i] = k_y[:]

    " Set Physical Parameters"

    boltzman_evt = (1.38 * 10 ** -23) * temperature / (1.6 * 10 ** -19) # Convert into Boltzman eV units for ease
    Interaction_Potential_1, Interaction_Potential_2, pairing_x, pairing_y,  = gap_symmetry(gap_pairing, kx, ky)
    Iteration_Constant_xsq_ysq = (-Interaction_Potential_1 / (2 * k_space_n))  # Combined constant for the integral sum
    Iteration_Constant_xy = (- Interaction_Potential_2 / (2 * k_space_n))  # Combined constant for the integral sum


    "create arrays"

    Delta_Iteration_x = np.zeros((Delta_convergence, hopping_step))
    Delta_Iteration_y = np.zeros((Delta_convergence, hopping_step))
    epsilon_k_squared = epsilon_k ** 2 #Square to avoid loop
    Delta_Temperature_x = np.zeros((hopping_step, temperature_step))
    Delta_Temperature_y = np.zeros((hopping_step, temperature_step))
    Tc_x = np.zeros((hopping_step))
    Tc_y = np.zeros((hopping_step))
    Delta_pairing = np.zeros((n_x, n_y), dtype=complex) #Allow for complex order paramaters


    "Run calculation"
    for hopping in range(hopping_step):
        if hopping == 0:
            Delta_Iteration_x[0, 0], Delta_Iteration_y[0, 0] = Delta_guess, Delta_guess
        else:
            # Tc and delta vary smoothly, thus this assumption saves convergence time
            Delta_Iteration_x[0, 0], Delta_Iteration_y[0, 0] = Delta_Temperature_x[hopping - 1, 0], Delta_Temperature_y[hopping - 1, 0]

        for temp in range(temperature_step):

            for n in range(1, Delta_convergence):

                Delta_x = Delta_Iteration_x[n - 1, 0] * pairing_x[:, :]
                Delta_y = 1j * Delta_Iteration_y[n - 1, 0] * pairing_y[:, :]
                Delta_pairing[:, :] = (Delta_x[:, :] + Delta_y[:, :])
                Energy_k = np.sqrt(epsilon_k_squared[hopping, :, :] + np.abs(Delta_pairing) ** 2)
                Fermi_dirac = 1 / (np.exp(Energy_k / boltzman_evt[temp]) + 1)

                Delta_Sum_x = delta_sum(Iteration_Constant_xsq_ysq, np.real(Delta_pairing), Fermi_dirac, Energy_k, pairing_x)
                Delta_Sum_y = delta_sum(Iteration_Constant_xy, np.imag(Delta_pairing), Fermi_dirac, Energy_k, pairing_y)
                Delta_Iteration_x[n, 0] = (1 - mixing) * Delta_Iteration_x[n - 1, 0] + (mixing * Delta_Sum_x)
                Delta_Iteration_y[n, 0] = (1 - mixing) * Delta_Iteration_y[n - 1, 0] + (mixing * Delta_Sum_y)
                if (np.absolute(Delta_Iteration_x[n, 0] - Delta_Iteration_x[n - 1, 0]) + np.absolute(
                        Delta_Iteration_y[n, 0] - Delta_Iteration_y[n - 1, 0])) < 50 * tolerance:
                    Delta_Temperature_x[hopping, temp] = Delta_Iteration_x[n, 0]
                    Delta_Temperature_y[hopping, temp] = Delta_Iteration_y[n, 0]

                    if n == Delta_convergence:
                        print("NOT CONVERGED AT [t, T] =", hopping, temp)

                    break
            Delta_Iteration_x[0, 0], Delta_Iteration_y[0, 0] = Delta_Temperature_x[hopping, temp],  Delta_Temperature_y[hopping, temp]


    "Find Tc"
    for hopping in range(hopping_step):
        for temp in range(temperature_step):
            if Delta_Temperature_x[hopping, temp] / Delta_Temperature_x[hopping, 0] < 0.05:
                Tc_x[hopping] = temperature[temp]
                break

    for hopping in range(hopping_step):
        for temp in range(temperature_step):
            if Delta_Temperature_y[hopping, temp] / Delta_Temperature_y[hopping, 0] < 0.05:
                Tc_y[hopping] = temperature[temp]
                break

    return Delta_Temperature_x, Delta_Temperature_y, Tc_x, Tc_y, datetime.now() - startTime

#The list of strains: uniaxial, shear, c_axis. The list of gap_pairings is listed in function "gap_symmetry"
delta(strain="uniaxial", gap_pairing="s_wave", temperature_step=200, temperature_max=2, save=True, tc_plot=True)


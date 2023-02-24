import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve

startTime = datetime.now()

" Create functions "

def fermi_dirac_func(e, kt):
    return 1 / (np.exp(e/kt) + 1)



def e_k(strain, n_kx, n_ky, hopping_step, save):

    "Use this to solve for the change in fermi level st it is kept constant with strain"
    " It is derived from rearrange the equation for Number of electrons under the assumption this is kept const."
    def fermi_energy_solver(e):
        return Ne / 2 - np.sum(1 / (np.exp(((e + gamma_k[h, :, :]) / KbT)) + 1))

    "  Set Zero arrays to be written into "
    # gamma_k and epsilon_0 together make up epsilon_k, but separated for constant electron number
    epsilon_k = np.zeros((hopping_step, n_ky, n_kx))
    gamma_k = np.zeros((np.shape(epsilon_k)))
    epsilon_0 = np.zeros((1, hopping_step))


    " Set physical constants "

    epsilon_0[0, 0] = -131 * 10 ** -3 #initial onsite//fermi energy value
    t_x = 81.62 * 10 ** -3  # Hopping integral value xdir, nearest neighbour
    t_prime = 36.73 * 10 ** -3  # Second-nearest neighbour hopping integral
    KbT = 0 #Boltzman const * Temperature (if not zero, remember to convert to eV)
    hbar = 1.054 * 10 ** -34
    atomic_radii = 1.05 * 10 ** -10
    m_e = 9.11 * 10 ** -31 #electron mass
    interatomic_prefactor = 9 * (hbar **2) * (atomic_radii ** (3/2)) / (m_e)

    "calculate hopping changes based on type of strain"
    if strain == "c_axis":
       t_x_array = np.linspace(1 * t_x, 1.43 * t_x, hopping_step)
    if strain == "uniaxial":
        t_x_array = np.linspace(1 * t_x, 1.14 * t_x, hopping_step)  # Hopping integral linear change array
        V_xy = -0.42  # Poisson ratio (approx for Sr2RuO4)
        t_y_array = (V_xy * (t_x_array[:] - t_x)) + t_x
        
    if strain == 'shear':
        t_xy_array = np.zeros((1, hopping_step))
        xy_0 = (interatomic_prefactor / t_prime) ** (2 / 7)
        x_0 = (interatomic_prefactor / t_x) ** (2 / 7)
        dx = np.linspace(0, +0.135, hopping_step) * x_0  # topological change at 0.113

    " set up k space in first Brilluion zone   "
    bz_edge = np.pi
    k_x = np.linspace(-bz_edge, bz_edge, n_kx)
    k_y = np.linspace(+bz_edge, -bz_edge, n_ky)
    kx = np.zeros((n_kx, n_ky))
    ky = np.zeros((np.shape(kx)))

    " Use kx and ky to multiply trig functions - this avoids extra for loops"
    for i in range(n_kx):
        kx[i, :] = k_x[:]
        ky[:, i] = k_y[:]
    
    "calculate The number of electrons - Ne"
    e_k_ne = epsilon_0[0, 0] - (2 * t_x * (np.cos(ky[:, :]) + np.cos(kx[:, :]))) - 4 * t_prime * (
            np.cos(kx[:, :]) * np.cos(ky[:, :]))
    Ne = 2 * np.sum(fermi_dirac_func(e_k_ne, KbT))
    
    " Create an epsilon_k for every combination of (k_x,k_y) points at each strain "
    for h in range(hopping_step):
        if strain == "uniaxial":
            gamma_k[h, :, :] = - 2 * (
                        t_x_array[h] * np.cos(kx[:, :]) + t_y_array[h] * np.cos(ky[:, :])) - 4 * t_prime * (
                                           np.cos(kx[:, :]) * np.cos(ky[:, :]))
        if strain == "c_axis":
            gamma_k[h, :, :] = (-2 * t_x_array[h] * (np.cos(kx[:, :]) + np.cos(ky[:, :]))) - (4 * t_prime *
                                  np.cos(kx[:, :]) * np.cos([ky[:, :]]))
        if strain == "shear":
            t_x_array[h] = interatomic_prefactor * (np.sqrt((x_0 - dx[h]) ** 2 + (dx[h] ** 2))) ** (-7 / 2)
            t_xy_array[h] = interatomic_prefactor * (xy_0 - 2 * np.sqrt(2) * dx[h]) ** (-7 / 2)
            gamma_k[h, :, :] = -2 * (t_x_array[h] * (np.cos(kx[:, :]) + np.cos(ky[:, :])) + (t_xy_array[h] + t_prime) *
                                     np.cos(kx[:, :]) * np.cos([ky[:, :]]) + (t_xy_array[h] - t_prime) * np.sin(
                        kx[:, :])
                                     * np.sin(ky[:, :]))

        "combine above 'gamma' with a strain-solved epsilon_0, using the intial value to calculate the number of electrons"
        "This keep the number of electrons constant by fixing Ne and readjusting the fermi-level inlcuded in epsilon_0"
        "This is an assumption that no longer holds if another band is used"

        
        if h == 0:
            epsilon_k[h, :, :] = epsilon_0[0, 0] + gamma_k[h, :, :]
        else:
            epsilon_0[0, h] = fsolve(fermi_energy_solver, epsilon_0[0, h - 1])
            epsilon_k[h, :, :] = epsilon_0[0, h] + gamma_k[h, :, :]


    print('code runtime = ', datetime.now() - startTime)


    " Print Variables "
    if save:
        np.save("Energyband_{}".format(strain), epsilon_k)
        np.save("hopping_array_{}".format(strain), t_x_array)
        np.save("epsilon_0_{}".format(strain), epsilon_0)


    "Plot first and last strains to check it is as expected"
    fig, ax = plt.subplots(figsize=(6, 6))
    cntr1 = ax.contour(k_x, k_y, epsilon_k[0, :, :], colors='red')
    cntr2 = ax.contour(k_x, k_y, epsilon_k[-1, :, :], colors="blue")
    ax.set_xlabel("$k_x$", fontsize=20)
    ax.set_ylabel("$k_y$", fontsize=20)
    ax.set_title("Energyband: strain = {}".format(strain), fontsize=20)
    ax.set_xticks((-bz_edge, 0, bz_edge), ['$-\pi$', '0', '$\pi$'])
    ax.set_yticks((bz_edge, 0, -bz_edge), ['$\pi$', '0', '$-\pi$'])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    h1, _ = cntr1.legend_elements()
    h2, _ = cntr2.legend_elements()
    number = t_x_array[-1] / t_x
    ax.legend([h1[0], h2[0]], ['Unstrained', 'strain = {}'.format(number)])


"e_k(kx,ky, hopping step)"
e_k(strain="uniaxial", n_kx=700, n_ky=700, hopping_step=20, save=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot(strain, gap_pairing, plot_dos, plot_tc, tc_inset):
    "Calls plotting functions and assigns the outputs then saves"

    if plot_dos:
        ax1, ax2 = dos_plot(strain, gap_pairing)
        ax1.set_title("{}".format(gap_pairing), fontsize=32)
        plt.savefig("{}_{}_dos_weighted.jpg".format(gap_pairing, strain))

    if plot_tc:
        ax1 = tc_plot(strain, gap_pairing, tc_inset)
        ax1.set_title("{}".format(gap_pairing), fontsize=32)
        plt.savefig("{}_tc_{}.jpg".format(gap_pairing, strain))




def dos_plot(strain, gap_pairing):
    "Returns plots for dos"

    "Load variables"
    dos = np.load("{}_{}_dos.npy".format(gap_pairing, strain))
    fermi_surface = np.load("{}_{}_fermi_surface.npy".format(gap_pairing, strain))
    hopping_array = np.load("hopping_array_{}.npy".format(strain))

    "create physical arrays and assign array size based on input arrays "
    hopping_step, dos_step = np.shape(dos)
    n_x, n_y = np.shape(fermi_surface[0, :, :])
    t_x = 81.62 * 10 ** -3
    epsilon = np.linspace(-2 * t_x, 2.5 * t_x,
                          dos_step) * 10 ** 3  # This is hard coded in for now, *MUST* match epsilon used for dos calculation
    k_x = np.linspace(-1 * np.pi, +1 * np.pi, n_x)
    k_y = np.linspace(+1 * np.pi, -1 * np.pi, n_y)

    "chose which slice of 3D array in the strain direction to display the dos"
    n_1 = 12
    n_2 = 16
    t_x_1 = round(hopping_array[n_1] / hopping_array[0], 2)  # Converts code number into a strain
    t_x_2 = round(hopping_array[n_2] / hopping_array[0], 2)
    t_x_end = round(hopping_array[-1] / hopping_array[0], 2)

    " Create figure and axis sizes "
    fig, ax1 = plt.subplots(figsize=(8, 8))
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.15, 0.61, 0.225, 0.225]
    ax2 = fig.add_axes([left, bottom, width, height])

    "Plot ax1, dos vs energy at various strains"
    ax1.plot(epsilon[:], dos[0, :], 'b.', label=(hopping_array[0] / hopping_array[0]))
    ax1.plot(epsilon[:], dos[n_1, :], 'g.', label=(t_x_1))
    ax1.plot(epsilon[:], dos[n_2, :], 'r.', label=(t_x_2))
    ax1.plot(epsilon[:], dos[-1, :], 'm.', label=(t_x_end))
    ax1.set_xticks((-t_x * 10 ** 3, 0, t_x * 10 ** 3), ['$-t_0$', '$\epsilon_{f}$', '$t_0$'])
    ax1.xaxis.set_tick_params(labelsize=28)
    ax1.set_xlabel("$\\varepsilon$ / meV", fontsize=32)
    ax1.set_ylabel("$\\rho(\\varepsilon)$", fontsize=36)
    ax1.legend(loc='upper right', title='$\\frac{t_x}{t_0}$', prop={'size': 24}, title_fontsize=28, markerscale=4)

    "ax2 plot, Fermi-surface inset"

    ax2.contour(k_x, k_y, fermi_surface[0, :, :], colors='blue')
    ax2.contour(k_x, k_y, fermi_surface[1, :, :], colors='red')
    ax2.xaxis.set_tick_params(labelsize=15, labeltop=True, labelbottom=False)
    ax2.yaxis.set_tick_params(labelsize=15, labelright=True, labelleft=False)
    ax2.set_xticks((-np.pi, 0, np.pi), ['$-\pi$', '0', '$\pi$'])
    ax2.set_yticks((np.pi, 0, -np.pi), ['$\pi$', '0', '$-\pi$'])

    return ax1, ax2

def tc_plot(strain, gap_pairing, tc_inset):
    "returns plot for tc"

    "Determine gap type"
    if gap_pairing == "s_wave" or gap_pairing == "ext_s_wave" or gap_pairing == "d_wave":
        gap_type = "one component"
    else:
        gap_type = "two component"

    "Load variables"

    hopping_array = np.load("hopping_array_{}.npy".format(strain))

    if gap_type == "one component":

        tc = np.load("{}_tc_{}.npy".format(gap_pairing, strain))

        " Create figure and axis sizes "
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(hopping_array[:] / hopping_array[0], tc[:], 'r.', markersize=16)
        ax1.xaxis.set_tick_params(labelsize=15)
        ax1.yaxis.set_tick_params(labelsize=19)
        ax1.set_ylabel("Tc (k)", fontsize=32)
        ax1.set_xlabel("$t_x / t_0$", fontsize=32)
        ax1.locator_params(axis='y', nbins=8)

        return ax1


    if gap_type == "two component":

        tc_x = np.load("{}_tc_x_{}".format(gap_pairing, strain))
        tc_y = np.load("{}_tc_y_{}".format(gap_pairing, strain))

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(hopping_array[:] / hopping_array[0], tc_x[:], 'r-', markersize=16)
        ax1.plot(hopping_array[:] / hopping_array[0], tc_y[:], 'b.', markersize=16)
        ax1.xaxis.set_tick_params(labelsize=15)
        ax1.yaxis.set_tick_params(labelsize=19)
        ax1.set_ylabel("Tc (k)", fontsize=32)
        ax1.set_xlabel("$t_x / t_0$", fontsize=32)

        if tc_inset:
            delta_x = np.load("{}_delta_x_{}".format(gap_pairing, strain))
            delta_y = np.load("{}_delta_y_{}".format(gap_pairing, strain))
            # These are in unitless percentages of the figure size. (0,0 is bottom left)
            left, bottom, width, height = [0.75, 0.45, 0.15, 0.3]
            ax2 = fig.add_axes([left, bottom, width, height])
            ax2.plot(hopping_array[:] / hopping_array[0], delta_x[:, 0] * 10 ** 3, 'r.', markersize=10)
            ax2.plot(hopping_array[:] / hopping_array[0], delta_y[:, 0] * 10 ** 3, 'b.', markersize=10)
            ax2.set_ylabel("$\Delta(T=0)$ (meV)", fontsize=14)
            ax2.xaxis.set_tick_params(labelsize=10)
            ax2.yaxis.set_tick_params(labelsize=10)
            return ax1, ax2
        else:
            return ax1


plot(strain="uniaxial", gap_pairing="s_wave", plot_dos=True, plot_tc=True, tc_inset=False)
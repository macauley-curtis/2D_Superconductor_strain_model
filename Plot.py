import numpy as np
import matplotlib.pyplot as plt

def gap_titles(gap_pairing):
    "Returns the interaction potential(s) and the gap pairing(s)"
    if gap_pairing == "s_wave":
        return "s-wave", "one component", 0, 0
    if gap_pairing == "d_wave":
        return "d$_{x^2-y^2}$", "one component", 0, 0
    if gap_pairing == "ext_s_wave":
        return "s$^{\pm}$", "one component", 0, 0
    if gap_pairing == "d_plus_id":
        return "d+id", "two component", "$d_{x^2-y^2}$", "$d_{xy}$"
    if gap_pairing == "d_plus_ig":
        return "d+ig", "two component", "$d_{x^2-y^2}$", "g"
    if gap_pairing == "chiral_p":
        return "p$_x$ + ip$_y$", "two component", "p$_x$", "p$_y$"


def plot(strain, gap_pairing, plot_dos, plot_tc, tc_inset):
    "Calls plotting functions and assigns the outputs then saves"
    full_title, gap_type, x_title, y_title = gap_titles(gap_pairing)

    if gap_type == "one component":
        if plot_dos:
            ax1, ax2 = dos_plot(strain, gap_pairing)
            if gap_pairing == "s_wave":
                pass
            else:
                ax1.set_title("{}".format(full_title), fontsize=32)
            plt.savefig("{}_{}_dos_weighted.jpg".format(gap_pairing, strain))

        if plot_tc:
            ax1 = tc_plot(strain, gap_pairing, tc_inset)
            ax1.set_title("{}".format(full_title), fontsize=32)
            plt.savefig("{}_tc_{}.jpg".format(gap_pairing, strain))

    if gap_type == "two component":
        if plot_dos:
            ax1, ax2, ax3, ax4 = dos_plot(strain, gap_pairing)
            plt.figure(1)
            ax1.set_title("{}".format(x_title), fontsize=32)
            plt.savefig("{}_{}_x_dos_weighted.jpg".format(gap_pairing, strain))
            plt.figure(2)
            ax3.set_title("{}".format(y_title), fontsize=32)
            plt.savefig("{}_{}_y_dos_weighted.jpg".format(gap_pairing, strain))


        if plot_tc:
            ax1 = tc_plot(strain, gap_pairing, tc_inset)
            ax1.set_title("{}".format(full_title), fontsize=32)
            plt.savefig("{}_tc_{}.jpg".format(gap_pairing, strain))



def dos_plot(strain, gap_pairing):
    "Returns plots for dos"

    dummy_var, gap_type, dummy_var, dummy_var = gap_titles(gap_pairing)
    if gap_type == "one component":
        "Load variables"
        dos = np.load("{}_{}_dos.npy".format(gap_pairing, strain))
        fermi_surface = np.load("{}_{}_fermi_surface.npy".format(gap_pairing, strain))
        hopping_step, dos_step = np.shape(dos)
        n_x, n_y = np.shape(fermi_surface[0, :, :])

    if gap_type == "two component":
        dos_x = np.load("{}_x_dos_{}.npy".format(gap_pairing, strain))
        dos_y = np.load("{}_y_dos_{}.npy".format(gap_pairing, strain))
        fermi_surface_x = np.load("{}_x_fermi_surface_{}.npy".format(gap_pairing, strain))
        fermi_surface_y = np.load("{}_y_fermi_surface_{}.npy".format(gap_pairing, strain))
        hopping_step, dos_step = np.shape(dos_x)
        n_x, n_y = np.shape(fermi_surface_x[0, :, :])




    hopping_array = np.load("hopping_array_{}.npy".format(strain))

    "create physical arrays and assign array size based on input arrays "
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

    if gap_type == "one component":
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

    if gap_type == "two component":
        " Create figure and axis sizes "
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.635, 0.2, 0.225, 0.225]
        ax2 = fig1.add_axes([left, bottom, width, height])

        "Plot ax1, dos vs energy at various strains"
        ax1.plot(epsilon[:], dos_x[0, :], 'b.', label=(hopping_array[0] / hopping_array[0]))
        ax1.plot(epsilon[:], dos_x[n_1, :], 'g.', label=(t_x_1))
        ax1.plot(epsilon[:], dos_x[n_2, :], 'r.', label=(t_x_2))
        ax1.plot(epsilon[:], dos_x[-1, :], 'm.', label=(t_x_end))
        ax1.set_xticks((-t_x * 10 ** 3, 0, t_x * 10 ** 3), ['$-t_0$', '$\epsilon_{f}$', '$t_0$'])
        ax1.xaxis.set_tick_params(labelsize=28)
        ax1.set_xlabel("$\\varepsilon$ / meV", fontsize=32)
        ax1.set_ylabel("$\\rho(\\varepsilon)$", fontsize=36)
        ax1.legend(loc='upper right', title='$\\frac{t_x}{t_0}$', prop={'size': 24}, title_fontsize=28, markerscale=4)

        "ax2 plot, Fermi-surface inset"

        ax2.contour(k_x, k_y, fermi_surface_x[0, :, :], colors='blue')
        ax2.contour(k_x, k_y, fermi_surface_x[1, :, :], colors='red')
        ax2.xaxis.set_tick_params(labelsize=15, labeltop=True, labelbottom=False)
        ax2.yaxis.set_tick_params(labelsize=15, labelright=True, labelleft=False)
        ax2.set_xticks((-np.pi, 0, np.pi), ['$-\pi$', '0', '$\pi$'])
        ax2.set_yticks((np.pi, 0, -np.pi), ['$\pi$', '0', '$-\pi$'])

        " Create figure and axis sizes "
        fig2, ax3 = plt.subplots(figsize=(8, 8))
        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.635, 0.2, 0.225, 0.225]
        ax4 = fig2.add_axes([left, bottom, width, height])

        "Plot ax1, dos vs energy at various strains"
        ax3.plot(epsilon[:], dos_y[0, :], 'b.', label=(hopping_array[0] / hopping_array[0]))
        ax3.plot(epsilon[:], dos_y[n_1, :], 'g.', label=(t_x_1))
        ax3.plot(epsilon[:], dos_y[n_2, :], 'r.', label=(t_x_2))
        ax3.plot(epsilon[:], dos_y[-1, :], 'm.', label=(t_x_end))
        ax3.set_xticks((-t_x * 10 ** 3, 0, t_x * 10 ** 3), ['$-t_0$', '$\epsilon_{f}$', '$t_0$'])
        ax3.xaxis.set_tick_params(labelsize=28)
        ax3.set_xlabel("$\\varepsilon$ / meV", fontsize=32)
        ax3.set_ylabel("$\\rho(\\varepsilon)$", fontsize=36)
        ax3.legend(loc='upper right', title='$\\frac{t_x}{t_0}$', prop={'size': 24}, title_fontsize=28, markerscale=4)

        "ax2 plot, Fermi-surface inset"

        ax4.contour(k_x, k_y, fermi_surface_y[0, :, :], colors='blue')
        ax4.contour(k_x, k_y, fermi_surface_y[1, :, :], colors='red')
        ax4.xaxis.set_tick_params(labelsize=15, labeltop=True, labelbottom=False)
        ax4.yaxis.set_tick_params(labelsize=15, labelright=True, labelleft=False)
        ax4.set_xticks((-np.pi, 0, np.pi), ['$-\pi$', '0', '$\pi$'])
        ax4.set_yticks((np.pi, 0, -np.pi), ['$\pi$', '0', '$-\pi$'])

        return ax1, ax2, ax3, ax4

def tc_plot(strain, gap_pairing, tc_inset):
    "returns plot for tc"

    "call title and gap type"
    title, gap_type, dummy_var, dummy_var = gap_titles(gap_pairing)

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

        tc_x = np.load("{}_tc_x_{}.npy".format(gap_pairing, strain))
        tc_y = np.load("{}_tc_y_{}.npy".format(gap_pairing, strain))

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(hopping_array[:] / hopping_array[0], tc_x[:], 'r.', markersize=16)
        ax1.plot(hopping_array[:] / hopping_array[0], tc_y[:], 'b.', markersize=16)
        ax1.xaxis.set_tick_params(labelsize=15)
        ax1.yaxis.set_tick_params(labelsize=19)
        ax1.set_ylabel("Tc (k)", fontsize=32)
        ax1.set_xlabel("$t_x / t_0$", fontsize=32)

        if tc_inset:
            delta_x = np.load("{}_delta_x_{}.npy".format(gap_pairing, strain))
            delta_y = np.load("{}_delta_y_{}.npy".format(gap_pairing, strain))
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

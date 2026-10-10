import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src import fetch_gap_titles
from src.utils.cli_output import toast
from src.utils.helpers import generate_plot_path


def plot_tc(tc, hopping_array, params):
    """
    Plot the critical temperature as a function of strain.

    :param tc: Critical temperature value.
    :param hopping_array: Array of hoppinh values.
    :param strain: Type of strain (e.g., 'uniaxial', 'c_axis', 'shear').
    :param gap_pairing: Type of gap pairing (e.g., 's_wave', 'd_wave').
    """
    
    # file path for saving the plot
    plot_file = generate_plot_path(
        params.strain, params.gap_pairing, "tc", extension="png"
        ) 
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    
    if np.shape(tc) == (1, len(hopping_array)):
        _gap_title, component_title = fetch_gap_titles.fetch_gap_titles(
            params.gap_pairing
            )
        
        plt.plot(hopping_array[:, 0] / params.tx, tc[0,:], 'b.',
                label=f'Strain: {params.strain}, Gap Pairing: {component_title}',
                markersize=8)
    else:
        _gap_title,_n_comp,comp_1,comp_2 = fetch_gap_titles.fetch_gap_titles(
            params.gap_pairing
            )
        plt.plot(hopping_array[:, 0] / params.tx, tc[0, :, 0], 'b.', label=f'{comp_1}', markersize=8)
        plt.plot(hopping_array[:, 0] / params.tx, tc[0, :, 1], 'r.', label=f'{comp_2}', markersize=8)
     
        
    ax.set_xlabel("$t_x$ / $t_0$", fontsize=14)
    ax.set_ylabel("Tc (K)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Critical Temperature vs Strain", fontsize=16)
    ax.legend()

    fig.savefig(plot_file)
    plt.close(fig)
    toast(
        f"Critical temperature plot saved to: {plot_file}",
        color="green",
        title="Plot Saved",
    )
    
        
def plot_delta(
    params,
    delta,
    temperature,
    hopping_array,
    kx,
    ky,
    fps=2,
):
    """
    Create a movie showing Delta(T) for each hopping step.

    Parameters
    ----------
    delta : ndarray
        Delta array with shape:

            (hopping_step, temperature_step, components)

        For a one-component gap:
            delta[:, :, 0]

        For a two-component gap:
            delta[:, :, 0]
            delta[:, :, 1]

    temperature : ndarray
        Temperature values in K.

    hopping_array : ndarray
        Hopping values corresponding to the first dimension of delta.

    fps : int
        Frames per second.
    """

    n_components = delta.shape[2]

    # Get gap title and component titles
    if n_components == 1:
        gap_title, component_title = fetch_gap_titles.fetch_gap_titles(
            params.gap_pairing
        )
    else:
        gap_title, _n_comp, comp_1, comp_2 = fetch_gap_titles.fetch_gap_titles(
            params.gap_pairing
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert to meV
    delta = delta * 10**3

    delta_min = np.min(delta)
    delta_max = np.max(delta)

    y_padding = 0.05 * (delta_max - delta_min)

    if y_padding == 0:
        y_padding = 1.0

    ax.set_xlim(
        np.min(temperature),
        np.max(temperature)
    )

    ax.set_ylim(
        delta_min - y_padding,
        delta_max + y_padding
    )

    ax.set_xlabel("Temperature (K)", fontsize=14)
    ax.set_ylabel(r"$\Delta$ (meV)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    # --------------------------------------------------
    # Set component labels
    # --------------------------------------------------

    if n_components == 1:
        component_labels = [component_title]

    elif n_components == 2:
        component_labels = [comp_1, comp_2]

    else:
        component_labels = [
            f"Component {i + 1}"
            for i in range(n_components)
        ]

    # Create lines
    lines = []

    for component in range(n_components):

        line, = ax.plot(
            [],
            [],
            label=component_labels[component],
            linewidth=2
        )

        lines.append(line)

    ax.legend()

    def update(frame):

        for component, line in enumerate(lines):

            line.set_data(
                temperature,
                delta[frame, :, component]
            )

        hopping_value = hopping_array[frame, 0]

        ax.set_title(
            f"Strain: {params.strain}, "
            f"Gap: {gap_title}\n"
            f"Hopping = {(hopping_value * 10**3):.2f} meV "
            f"({hopping_value / params.tx:.2f} $t_x$)"
        )

        return lines

    animation = FuncAnimation(
        fig,
        update,
        frames=len(hopping_array),
        interval=1000 / fps,
        blit=True
    )

    animation.save(
        generate_plot_path(
            params.strain,
            params.gap_pairing,
            "delta_vs_temperature",
            extension="mp4"
        ),
        writer="ffmpeg",
        fps=fps
    )

    plt.close(fig)

    toast(
        f"Delta(T) animation saved to: "
        f"{generate_plot_path(params.strain, params.gap_pairing, 'delta_vs_temperature', extension='mp4')}",
        color="green",
        title="Animation Saved",
    )

def plot_density_of_states(
    params,
    dos,
    epsilon,
    hopping_array,
    n_components,
    normalize=True,
    fps=2,
):
    """
    Create an MP4 animation of the density of states versus energy
    for each hopping step.

    Parameters
    ----------
    params : object
        Model parameters, including strain and gap_pairing.
    dos : ndarray
        Density of states with shape
        (hopping_step, energy_step, n_components).
    epsilon : ndarray
        Energy values.
    hopping_array : ndarray
        Hopping parameters with shape (hopping_step, 2).
    n_components : int
        Number of gap-function components.
    normalize : bool
        Whether to normalize each component at each hopping step.
    fps : int
        Frames per second.
    """

    dos = np.asarray(dos)
    epsilon = np.asarray(epsilon)
    hopping_array = np.asarray(hopping_array)

    # Validate array dimensions
    if dos.ndim != 3:
        raise ValueError(
            "dos must have shape "
            "(hopping_step, energy_step, n_components), "
            f"but received shape {dos.shape}."
        )

    if epsilon.ndim != 1:
        raise ValueError(
            f"epsilon must be 1D, but received shape {epsilon.shape}."
        )

    if dos.shape[1] != len(epsilon):
        raise ValueError(
            f"dos has {dos.shape[1]} energy points, "
            f"but epsilon has {len(epsilon)}."
        )

    if dos.shape[2] != n_components:
        raise ValueError(
            f"dos has {dos.shape[2]} components, "
            f"but n_components is {n_components}."
        )

    if hopping_array.shape != (dos.shape[0], 2):
        raise ValueError(
            f"hopping_array must have shape ({dos.shape[0]}, 2), "
            f"but received shape {hopping_array.shape}."
        )

    # Normalize DOS if requested
    if normalize:
        max_dos = np.max(np.abs(dos), axis=1, keepdims=True)

        dos = np.divide(
            dos,
            max_dos,
            out=np.zeros_like(dos, dtype=float),
            where=max_dos != 0,
        )

    # Get gap title and component titles
    if n_components == 1:
        gap_title, component_title = (
            fetch_gap_titles.fetch_gap_titles(params.gap_pairing)
        )
        component_labels = [component_title]

    elif n_components == 2:
        gap_title, _n_comp, comp_1, comp_2 = (
            fetch_gap_titles.fetch_gap_titles(params.gap_pairing)
        )
        component_labels = [comp_1, comp_2]

    else:
        gap_title = params.gap_pairing
        component_labels = [
            f"Component {i + 1}"
            for i in range(n_components)
        ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    colours = ["b", "r", "g", "orange"]

    lines = []

    for component in range(n_components):
        line, = ax.plot(
            [],
            [],
            color=colours[component % len(colours)],
            linewidth=2,
            label=component_labels[component],
        )
        lines.append(line)

    # Set fixed axis limits
    ax.set_xlim(
        np.min(epsilon),
        np.max(epsilon),
    )

    dos_min = np.nanmin(dos)
    dos_max = np.nanmax(dos)

    y_padding = 0.05 * (dos_max - dos_min)

    if y_padding == 0:
        y_padding = 1.0

    ax.set_ylim(
        dos_min - y_padding,
        dos_max + y_padding,
    )

    # Axis styling to match plot_delta
    ax.set_xlabel(r"Energy $\epsilon$", fontsize=14)

    if normalize:
        ax.set_ylabel("Normalised DOS", fontsize=14)
    else:
        ax.set_ylabel("Density of states", fontsize=14)

    ax.tick_params(axis="both", labelsize=12)
    ax.legend()

    # Update animation for each hopping step
    def update(frame):

        for component, line in enumerate(lines):
            line.set_data(
                epsilon,
                dos[frame, :, component],
            )

        hopping_value = hopping_array[frame, 0]

        ax.set_title(
            f"Strain: {params.strain}, "
            f"Gap: {gap_title}\n"
            f"Hopping = {hopping_value * 10**3:.2f} meV "
            f"({hopping_value / params.tx:.2f} $t_x$)",
            fontsize=16
        )

        return lines

    animation = FuncAnimation(
        fig,
        update,
        frames=dos.shape[0],
        interval=1000 / fps,
        blit=False,
    )

    # Save animation
    output_path = generate_plot_path(
        params.strain,
        params.gap_pairing,
        "density_of_states",
        extension="mp4",
    )

    animation.save(
        output_path,
        writer="ffmpeg",
        fps=fps,
    )

    plt.close(fig)

    toast(
        f"Density of states animation saved to: {output_path}",
        color="green",
        title="Animation Saved",
    )


def plot_energy_band(
    params,
    energy_band,
    kx,
    ky,
    hopping_array,
    fps=2,
):
    """
    Create an MP4 animation of energy-band contour lines
    for each hopping step.

    energy_band shape: (hopping_step, n_kx, n_ky)
    kx, ky: 2D k-space grids from create_square_k_space().
    hopping_array shape: (hopping_step, 2)
    """

    energy_band = np.asarray(energy_band)
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    hopping_array = np.asarray(hopping_array)

    n_hopping = energy_band.shape[0]

    # --------------------------------------------------
    # Use the existing k-space grids
    # --------------------------------------------------

    KX = kx
    KY = ky

    # Downsample for faster plotting if the grid is large
    plot_stride = max(
        1,
        int(np.ceil(max(energy_band.shape[1:]) / 200))
    )

    KX_plot = KX[::plot_stride, ::plot_stride]
    KY_plot = KY[::plot_stride, ::plot_stride]

    # --------------------------------------------------
    # Set fixed contour levels for all frames
    # --------------------------------------------------

    energy_min = np.nanmin(energy_band)
    energy_max = np.nanmax(energy_band)

    if energy_min == energy_max:
        energy_min -= 0.5
        energy_max += 0.5

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlabel(r"$k_x$", fontsize=20)
    ax.set_ylabel(r"$k_y$", fontsize=20)

    ax.set_xlim(np.nanmin(KX), np.nanmax(KX))
    ax.set_ylim(np.nanmin(KY), np.nanmax(KY))
    ax.set_aspect("equal")

    # Match the style of your old plot
    ax.set_xticks(
        [-np.pi, 0, np.pi],
        [r"$-\pi$", "0", r"$\pi$"],
    )

    ax.set_yticks(
        [np.pi, 0, -np.pi],
        [r"$\pi$", "0", r"$-\pi$"],
    )

    ax.tick_params(axis="both", labelsize=16)

    contour = [None]

    # --------------------------------------------------
    # Update each frame
    # --------------------------------------------------

    def update(frame):

        # Remove the previous contour lines
        if contour[0] is not None:
            contour[0].remove()

        # Current energy dispersion
        energy = energy_band[
            frame,
            ::plot_stride,
            ::plot_stride,
        ]

        # Draw contour lines, like the original code
        contour[0] = ax.contour(
            KX_plot,
            KY_plot,
            energy,
            levels=[0.0],
            cmap="viridis",
            linewidths=1.2,
        )

        hopping_value = hopping_array[frame, 0]

        ax.set_title(
            rf"$\varepsilon_k$: {params.strain}" "\n"
            f"Hopping = {hopping_value * 1e3:.2f} meV "
            f"({hopping_value / params.tx:.2f} $t_x$)",
            fontsize=16,
        )

        return ()

    # --------------------------------------------------
    # Animate
    # --------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=n_hopping,
        interval=1000 / fps,
        blit=False,
        cache_frame_data=False,
    )

    # --------------------------------------------------
    # Save MP4
    # --------------------------------------------------

    output_path = generate_plot_path(
        params.strain,
        params.gap_pairing,
        "energy_band",
        extension="mp4",
    )

    try:
        animation.save(
            output_path,
            writer="ffmpeg",
            fps=fps,
            dpi=80,
        )
    finally:
        plt.close(fig)

    toast(
        f"Energy band animation saved to: {output_path}",
        color="green",
        title="Animation Saved",
    )
import numpy as np

from src.delta_sum import delta_sum
from src.fermi_dirac import fermi_dirac_func


def calculate_delta(
    strain,
    e_k,
    n_k,
    convergence_num,
    hopping_step,
    temperature,
    temperature_step,
    conv_tolerance,
    mixing,
    delta_guess,
    V,
    gap_function,
    Kb,
    energy_cutoff,
    n_components
):
    """
    Calculate the superconducting gap funciton as a funciton of temperature and
    strain.
    
    :param strain: Type of strain (e.g., 'uniaxial', 'c_axis', 'shear').
    :param e_k: Energy band array as a function of hopping and k-space.
    :param n_k: Number of k-points in each direction (assuming square grid).
    :param convergence_num: Maximum number of iterations for convergence.
    :param hopping_step: Number of hopping values to consider.
    :param temperature: Array of temperatures to evaluate the gap function.
    :param temperature_step: Number of temperature values to consider.
    :param conv_tolerance: Convergence tolerance for the gap function.
    :param mixing: Mixing parameter for iterative convergence.
    :param delta_guess: Initial guess for the gap function.
    :param V: Pairing interaction strength (can be a scalar or array for 
    multi-component).
    :param gap_function: Gap pairing function (e.g., s-wave, d-wave)
    :param Kb: Boltzmann constant.
    :param energy_cutoff: Energy cutoff for active k-points (None for no cutoff)
    this is done to reduce the number of k-point calculations focusing only
    on the superconduting gap region.
    
    :param n_components: Number of components in the gap function (1 or 2).
    
    :return: the gap function as a function of temperature and strain
    """

    # Create arrays

    KbT = Kb * temperature
    k_space_n = n_k ** 2

    delta_iteration = np.zeros(convergence_num)
    
    if n_components == 1:
        delta_temperature = np.zeros(
                (hopping_step, temperature_step, 1)
                )
                
        iteration_constant = -V / (2 * k_space_n)
    else:
        
        c_1 = -V[0] / (2 * k_space_n)
        c_2 = -V[1] / (2 * k_space_n)
        
        gap_pairing_1 = gap_function[0]
        gap_pairing_2 = gap_function[1]
        
        delta_temperature = np.zeros(
                            (hopping_step, temperature_step, 2)
                        )


    # Calculate the gap function as a function of
    # temperature and strain

    if n_components == 1:
        
        delta_iteration = np.zeros(convergence_num)

        for hopping in range(hopping_step):
            
            if energy_cutoff is None:

                mask = np.ones(
                    (n_k, n_k),
                    dtype=bool,
                )

            else:

                mask = (
                    np.abs(e_k[hopping])
                    < energy_cutoff
                )

            # -------------------------------------------------
            # Reduce arrays to active k-points
            # -------------------------------------------------

            epsilon_active = e_k[hopping][mask]

            pairing_active = gap_function[mask]

            # Initial guess for this hopping value
            if hopping == 0:
                delta_iteration[0] = delta_guess
            else:
                delta_iteration[0] = (
                    delta_temperature[hopping - 1, 0, 0]
                )

            for temp in range(temperature_step):

                converged = False

                for n in range(1, convergence_num):

                    delta_pairing = (
                        delta_iteration[n - 1]
                        * pairing_active 
                    )

                    energy_k = np.sqrt(
                        epsilon_active ** 2
                        + np.abs(delta_pairing) ** 2
                    )

                    # Fermi-Dirac calculation unchanged
                    fermi_dirac = fermi_dirac_func(energy_k, KbT[temp])

                    delta_value = delta_sum(
                        c=iteration_constant,
                        delta=delta_pairing,
                        fe=fermi_dirac,
                        ek=energy_k,
                        pairing=pairing_active
                    )

                    delta_iteration[n] = (
                        (1 - mixing)
                        * delta_iteration[n - 1]
                        + mixing * delta_value
                    )

                    # Check convergence
                    if (
                        np.absolute(
                            delta_iteration[n]
                            - delta_iteration[n - 1]
                        )
                    ) < conv_tolerance:

                        delta_temperature[
                            hopping, temp, 0
                        ] = delta_iteration[n]

                        converged = True
                        break

                # If we have exhausted the iterations without
                # converging, raise an error.
                if not converged:
                    raise ValueError(
                        "NOT CONVERGED AT [t, T] =",
                        hopping,
                        temp
                    )

                # Use the converged Delta from this temperature
                # as the initial guess for the next temperature.
                delta_iteration[0] = (
                    delta_temperature[hopping, temp, 0]
                )
    else:           
            delta_iteration = np.zeros(
            (convergence_num, 2)
        )

            for hopping in range(hopping_step):

                # -------------------------------------------------
                # Create energy-shell mask
                # -------------------------------------------------

                if energy_cutoff is None:

                    mask = np.ones(
                        (n_k, n_k),
                        dtype=bool,
                    )

                else:

                    mask = (
                        np.abs(e_k[hopping])
                        < energy_cutoff
                    )

                # -------------------------------------------------
                # Reduce arrays
                # -------------------------------------------------

                epsilon_active = e_k[hopping][mask]

                pairing_1_active = (
                    gap_pairing_1[mask]
                )

                pairing_2_active = (
                    gap_pairing_2[mask]
                )

                # -------------------------------------------------
                # Initial guess
                # -------------------------------------------------

                if hopping == 0:

                    delta_iteration[0, 0] = delta_guess
                    delta_iteration[0, 1] = delta_guess

                else:

                    delta_iteration[0, 0] = (
                        delta_temperature[
                            hopping - 1,
                            0,
                            0,
                        ]
                    )

                    delta_iteration[0, 1] = (
                        delta_temperature[
                            hopping - 1,
                            0,
                            1,
                        ]
                    )

                # -------------------------------------------------
                # Temperature loop
                # -------------------------------------------------

                for temp in range(temperature_step):

                    converged = False

                    for n in range(
                        1,
                        convergence_num,
                    ):

                        # -----------------------------------------
                        # Component 1
                        # -----------------------------------------

                        delta_1 = (
                            delta_iteration[n - 1, 0]
                            * pairing_1_active
                        )

                        # -----------------------------------------
                        # Component 2
                        # -----------------------------------------

                        delta_2 = (
                            1j
                            * delta_iteration[n - 1, 1]
                            * pairing_2_active
                        )

                        delta_pairing = (
                            delta_1 + delta_2
                        )

                        # -----------------------------------------
                        # Quasiparticle energy
                        # -----------------------------------------

                        energy_k = np.sqrt(
                            epsilon_active**2
                            + np.abs(delta_pairing)**2
                        )

                        # -----------------------------------------
                        # Fermi function
                        # -----------------------------------------

                        fermi_dirac = fermi_dirac_func(
                            energy_k,
                            KbT[temp],
                        )

                        # -----------------------------------------
                        # Component 1 gap sum
                        # -----------------------------------------

                        delta_value_1 = delta_sum(
                            c=c_1,
                            delta=np.real(delta_pairing),
                            fe=fermi_dirac,
                            ek=energy_k,
                            pairing=pairing_1_active,
                        )

                        # -----------------------------------------
                        # Component 2 gap sum
                        # -----------------------------------------

                        delta_value_2 = delta_sum(
                            c=c_2,
                            delta=np.imag(delta_pairing),
                            fe=fermi_dirac,
                            ek=energy_k,
                            pairing=pairing_2_active,
                        )

                        # -----------------------------------------
                        # Mixing
                        # -----------------------------------------

                        delta_iteration[n, 0] = (
                            (1 - mixing)
                            * delta_iteration[n - 1, 0]
                            + mixing
                            * delta_value_1
                        )

                        delta_iteration[n, 1] = (
                            (1 - mixing)
                            * delta_iteration[n - 1, 1]
                            + mixing
                            * delta_value_2
                        )

                        # -----------------------------------------
                        # Convergence
                        # -----------------------------------------

                        component_1_converged = (
                            np.abs(
                                delta_iteration[n, 0]
                                - delta_iteration[n - 1, 0]
                            )
                            < conv_tolerance
                        )

                        component_2_converged = (
                            np.abs(
                                delta_iteration[n, 1]
                                - delta_iteration[n - 1, 1]
                            )
                            < conv_tolerance
                        )

                        if (
                            component_1_converged
                            and component_2_converged
                        ):

                            delta_temperature[
                                hopping,
                                temp,
                                0,
                            ] = delta_iteration[n, 0]

                            delta_temperature[
                                hopping,
                                temp,
                                1,
                            ] = delta_iteration[n, 1]

                            converged = True

                            break

                    if not converged:

                        raise ValueError(
                            f"Not converged at "
                            f"hopping={hopping}, "
                            f"temperature={temp}"
                        )

                    # Current temperature becomes
                    # next temperature's initial guess.

                    delta_iteration[0, 0] = (
                        delta_temperature[
                            hopping,
                            temp,
                            0,
                        ]
                    )

                    delta_iteration[0, 1] = (
                        delta_temperature[
                            hopping,
                            temp,
                            1,
                        ]
                    )

        
    return delta_temperature
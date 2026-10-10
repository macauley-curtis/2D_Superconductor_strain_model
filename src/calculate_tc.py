import numpy as np


def calculate_tc(
    Delta,
    temperature,
    tolerance_ratio,
    hopping_step
):

    tc = np.full((1, hopping_step), np.nan)

    for hopping in range(hopping_step):

        delta_t = Delta[hopping, :]

        delta_0 = delta_t[0]

        tolerance = delta_0 * tolerance_ratio

        indices = np.where(delta_t <= tolerance)[0]

        if len(indices) > 0:

            tc[0, hopping] = temperature[indices[0]]

        else:

            print(
                f"Warning: Tc not found for hopping {hopping}."
            )

    return tc

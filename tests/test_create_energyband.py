import numpy as np
import pytest

from src.create_energyband import create_energyband


def energyband_parameters(strain="c_axis", hopping_step=1):
    """
    Create the parameters needed for create_energyband().
    """

    n_k = 9

    k_values = np.linspace(-np.pi, np.pi, n_k)
    kx, ky = np.meshgrid(k_values, k_values, indexing="xy") 

    return {
        "strain": strain,
        "n_k": n_k,
        "kx": kx,
        "ky": ky,
        "t_x": 0.08162,
        "t_prime": 0.03673,
        "e_0": 0.0,
        "hopping_step": hopping_step,
        "KbT": 0.001,
    }


def create_test_hopping_array(parameters):
    """
    Create a hopping array with the shape expected by
    create_energyband().

    """

    hopping_step = parameters["hopping_step"]
    t_x = parameters["t_x"]

    return np.column_stack(
        (
            np.full(hopping_step, t_x),
            np.full(hopping_step, t_x),
        )
    )


def test_energyband_has_correct_shape():
    """
    Test that create_energyband() returns an energy band
    with the expected shape.
    """

    parameters = energyband_parameters(
        strain="c_axis",
        hopping_step=1,
    )

    hopping_array = create_test_hopping_array(parameters)

    energyband = create_energyband(
        strain=parameters["strain"],
        n_k=parameters["n_k"],
        kx=parameters["kx"],
        ky=parameters["ky"],
        t_x=parameters["t_x"],
        t_prime=parameters["t_prime"],
        e_0=parameters["e_0"],
        hopping_step=parameters["hopping_step"],
        hopping_array=hopping_array,
        KbT=parameters["KbT"],
    )

    assert energyband.shape == (
        parameters["hopping_step"],
        parameters["n_k"],
        parameters["n_k"],
    )


@pytest.mark.parametrize(
    "strain",
    [
        "uniaxial",
        "c_axis",
        "shear",
    ],
)
def test_each_strain_returns_one_band_per_hopping_step(strain):
    """
    Test that each valid strain produces one energy band
    for every hopping step.
    """

    hopping_step = 3

    parameters = energyband_parameters(
        strain=strain,
        hopping_step=hopping_step,
    )

    hopping_array = create_test_hopping_array(parameters)

    energyband = create_energyband(
        strain=parameters["strain"],
        n_k=parameters["n_k"],
        kx=parameters["kx"],
        ky=parameters["ky"],
        t_x=parameters["t_x"],
        t_prime=parameters["t_prime"],
        e_0=parameters["e_0"],
        hopping_step=parameters["hopping_step"],
        hopping_array=hopping_array,
        KbT=parameters["KbT"],
    )

    assert energyband.shape == (
        hopping_step,
        parameters["n_k"],
        parameters["n_k"],
    )

    assert np.isfinite(energyband).all()


def test_energyband_changes_with_hopping():
    """
    Test that changing the hopping parameters changes
    the resulting energy band.
    """

    parameters = energyband_parameters(
        strain="c_axis",
        hopping_step=2,
    )

    hopping_array = np.array(
        [
            [parameters["t_x"], parameters["t_x"]],
            [parameters["t_x"] * 0.9, parameters["t_x"] * 1.1],
        ]
    )

    energyband = create_energyband(
        strain=parameters["strain"],
        n_k=parameters["n_k"],
        kx=parameters["kx"],
        ky=parameters["ky"],
        t_x=parameters["t_x"],
        t_prime=parameters["t_prime"],
        e_0=parameters["e_0"],
        hopping_step=parameters["hopping_step"],
        hopping_array=hopping_array,
        KbT=parameters["KbT"],
    )

    assert energyband.shape == (
        2,
        parameters["n_k"],
        parameters["n_k"],
    )

    assert not np.allclose(
        energyband[0],
        energyband[1],
    )



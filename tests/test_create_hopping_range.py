import numpy as np

from src.create_hopping_range import create_hopping_range


def test_uniaxial_hopping_range():
    hopping_array = create_hopping_range(
        strain="uniaxial",
        t_x=1.0,
        t_prime=1.0,
        t_1=1.0,
        t_2=2.0,
        hopping_step=5,
        V_xy=0.5,
    )

    assert hopping_array.shape == (5, 2)

    expected_t1 = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    expected_t2 = np.array([1.0, 1.125, 1.25, 1.375, 1.5])

    np.testing.assert_allclose(hopping_array[:, 0], expected_t1)
    np.testing.assert_allclose(hopping_array[:, 1], expected_t2)


def test_c_axis_hopping_range():
    hopping_array = create_hopping_range(
        "c_axis",
        1.0,
        1.0,
        1.0,
        2.0,
        5,
    )

    assert hopping_array.shape == (5, 2)

    expected_t1 = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    expected_t2 = np.array([1.0, 1.25, 1.5, 1.75, 2.0])

    np.testing.assert_allclose(hopping_array[:, 0], expected_t1)
    np.testing.assert_allclose(hopping_array[:, 1], expected_t2)


def test_shear_hopping_range():
    hopping_array = create_hopping_range(
        "shear",
        1.0,
        1.0,
        1.0,
        2.0,
        5,
    )

    assert hopping_array.shape == (5, 2)

    expected_t1 = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    expected_t2 = np.array([1.0, 0.9, 0.8, 0.7, 0.6])

    np.testing.assert_allclose(hopping_array[:, 0], expected_t1)
    np.testing.assert_allclose(hopping_array[:, 1], expected_t2)
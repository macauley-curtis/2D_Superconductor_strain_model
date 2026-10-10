import numpy as np

from src.create_square_k_space import create_square_k_space


def test_create_square_k_space():
    nk = 5
    BZ_edge = 1
    kx, ky, = create_square_k_space(nk, BZ_edge)

    # Check the shapes of the returned arrays
    assert kx.shape == (nk, nk)
    assert ky.shape == (nk, nk)

    # Check the values of kx_array and ky_array
 
    expected_kx = np.array([
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    ])

    expected_ky = np.array([
        [np.pi, np.pi, np.pi, np.pi, np.pi],
        [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2],
        [0, 0, 0, 0, 0],
        [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2],
        [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi],
    ])

    assert np.array_equal(kx, expected_kx)
    assert np.array_equal(ky, expected_ky)
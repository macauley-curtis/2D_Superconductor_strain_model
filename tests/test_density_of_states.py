import numpy as np

from src.density_of_states import dos


def test_dos_simple():
    """Test DOS calculation for a single gap component."""

    gap_pairing = np.array([[2.0]])
    e_k = np.array([[[1.0]]])
    epsilon = np.array([1.0])

    result = dos(
        gap_pairing=gap_pairing,
        e_k=e_k,
        epsilon=epsilon,
        dos_step=1,
        screening_energy=1.0,
        hopping_step=1,
        nk=1,
        n_components=1,
    )

    # Expected DOS:
    # (1 / (nk**2 * pi)) * delta(1.0, 1.0 - 1.0) * 2.0**2
    expected = np.array([[[4 / np.pi]]])

    assert result.shape == (1, 1, 1)
    np.testing.assert_allclose(result, expected)

def test_dos_two_components():
    """Test DOS calculation for two gap components."""

    gap_pairing = np.array([[[2.0]], [[3.0]]])
    e_k = np.array([[[1.0]]])
    epsilon = np.array([1.0])

    result = dos(
        gap_pairing=gap_pairing,
        e_k=e_k,
        epsilon=epsilon,
        dos_step=1,
        screening_energy=1.0,
        hopping_step=1,
        nk=1,
        n_components=2,
    )

    expected = np.array([[[4 / np.pi, 9 / np.pi]]])

    assert result.shape == (1, 1, 2)
    np.testing.assert_allclose(result, expected)    
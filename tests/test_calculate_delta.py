import numpy as np

from src.calculate_delta import calculate_delta

# ============================================================
# Test k-space
# ============================================================

nk = 4

kx = np.array([
    [-np.pi, -np.pi / 3, np.pi / 3, np.pi],
    [-np.pi, -np.pi / 3, np.pi / 3, np.pi],
    [-np.pi, -np.pi / 3, np.pi / 3, np.pi],
    [-np.pi, -np.pi / 3, np.pi / 3, np.pi],
])

ky = np.array([
    [-np.pi, -np.pi, -np.pi, -np.pi],
    [-np.pi / 3, -np.pi / 3, -np.pi / 3, -np.pi / 3],
    [np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3],
    [np.pi, np.pi, np.pi, np.pi],
])


# ============================================================
# One-component gap
# ============================================================

def test_calculate_delta_one_component():
    """Test that the one-component calculation returns the expected shape."""

    hopping_step = 1
    temperature_step = 3

    temperature = np.array([
        0.1,
        0.2,
        0.3,
    ])

    # Simple flat energy band.
    e_k = np.ones(
        (hopping_step, nk, nk)
    ) * 0.05

    result = calculate_delta(
        strain="uniaxial",
        e_k=e_k,
        n_k=nk,
        convergence_num=100,
        hopping_step=hopping_step,
        temperature=temperature,
        temperature_step=temperature_step,
        conv_tolerance=1e-10,
        mixing=0.5,
        delta_guess=0.001,
        V=0.1,
        gap_function=np.ones((nk, nk)),
        Kb=8.617e-5,
        energy_cutoff=None,
        n_components=1
    )

    # One-component result should be:
    # (hopping_step, temperature_step, 1)
    assert result.shape == (1, 3, 1)

    # The calculation should produce finite values.
    assert np.all(np.isfinite(result))

    # The gap should not be negative.
    assert np.all(result >= 0.0)


# ============================================================
# Two-component gap
# ============================================================

def test_calculate_delta_two_component():
    """Test the two-component gap calculation."""

    hopping_step = 1
    temperature_step = 3

    temperature = np.array([
        0.1,
        0.2,
        0.3,
    ])

    # Simple flat energy band.
    e_k = np.full(
        (hopping_step, nk, nk),
        0.05,
    )

    # Two-component d + id gap:
    # g1(k) = cos(kx) - cos(ky)
    # g2(k) = sin(kx) * sin(ky)
    gap_function = np.stack(
        (
            np.cos(kx) - np.cos(ky),
            np.sin(kx) * np.sin(ky),
        ),
        axis=0,
    )

    result = calculate_delta(
        strain="uniaxial",
        e_k=e_k,
        n_k=nk,
        convergence_num=1000,
        hopping_step=hopping_step,
        temperature=temperature,
        temperature_step=temperature_step,
        conv_tolerance=1e-8,
        mixing=0.1,
        delta_guess=0.001,
        V=np.array([0.5, 0.3]),
        gap_function=gap_function,
        Kb=8.617e-5,
        energy_cutoff=None,
        n_components=2,
    )

    # Expected shape: (hopping_step, temperature_step, 2)
    assert result.shape == (1, 3, 2)

    # All values should be finite and non-negative.
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)

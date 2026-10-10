import numpy as np

from src.generate_gap_params import generate_gap_parameters

# ============================================================
# Test k-space
# ============================================================

nk= 3

kx = np.array([
    [-np.pi, 0.0, np.pi],
    [-np.pi, 0.0, np.pi],
    [-np.pi, 0.0, np.pi],
])

ky = np.array([
    [-np.pi, -np.pi, -np.pi],
    [0.0, 0.0, 0.0],
    [np.pi, np.pi, np.pi],
])


# ============================================================
# s-wave
# ============================================================

def test_s_wave_perpendicular():

    V, gap = generate_gap_parameters(
        gap_pairing="s_wave",
        strain="uniaxial",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected = np.ones((3, 3))

    assert V == -0.035

    np.testing.assert_array_equal(
        gap,
        expected,
    )


def test_s_wave_shear():

    V, gap = generate_gap_parameters(
        gap_pairing="s_wave",
        strain="shear",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected = np.ones((3, 3))

    assert V == -0.035

    np.testing.assert_array_equal(
        gap,
        expected,
    )


# ============================================================
# d-wave
# ============================================================

def test_d_wave_perpendicular():

    V, gap = generate_gap_parameters(
        gap_pairing="d_wave",
        strain="uniaxial",
        a1x=1.0,
        a1y=1.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected = np.array([
        [0, 2, 0],
        [-2, 0, -2],
        [0, 2, 0],
    ])

    assert V == -0.0394 * 0.5

    np.testing.assert_allclose(
        gap,
        expected,
    )


def test_d_wave_shear():

    V, gap = generate_gap_parameters(
        gap_pairing="d_wave",
        strain="shear",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected = np.array([
        [0, 2, 0],
        [-2, 0, -2],
        [0, 2, 0],
    ])

    assert V == -0.0394 * 0.5

    np.testing.assert_allclose(
        gap,
        expected,
    )


# ============================================================
# extended s-wave
# ============================================================

def test_ext_s_wave_perpendicular():

    V, gap = generate_gap_parameters(
        gap_pairing="ext_s_wave",
        strain="uniaxial",
        a1x=1.0,
        a1y=1.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected = np.array([
        [-2, 0, -2],
        [0, 2, 0],
        [-2, 0, -2],
    ])

    assert V == -0.06483

    np.testing.assert_allclose(
        gap,
        expected,
    )


def test_ext_s_wave_shear():

    V, gap = generate_gap_parameters(
        gap_pairing="ext_s_wave",
        strain="shear",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected = np.array([
        [-2, 0, -2],
        [0, 2, 0],
        [-2, 0, -2],
    ])

    assert V == -0.06483

    np.testing.assert_allclose(
        gap,
        expected,
    )


# ============================================================
# d + i d
# ============================================================

def test_d_plus_id_perpendicular():

    V, gap = generate_gap_parameters(
        gap_pairing="d_plus_id",
        strain="uniaxial",
        a1x=1.0,
        a1y=1.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected_d = np.array([
        [0, 2, 0],
        [-2, 0, -2],
        [0, 2, 0],
    ])

    expected_id = np.zeros((3, 3))

    expected_V = [
        -0.5 * 0.03915,
        -0.5 * 0.323549,
    ]

    assert V == expected_V

    np.testing.assert_allclose(
        gap[0],
        expected_d,
    )

    np.testing.assert_allclose(
        gap[1],
        expected_id,
        atol=1e-12,
    )


def test_d_plus_id_shear():

    V, gap = generate_gap_parameters(
        gap_pairing="d_plus_id",
        strain="shear",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected_d = np.array([
        [0, 2, 0],
        [-2, 0, -2],
        [0, 2, 0],
    ])

    expected_id = np.zeros((3, 3))

    expected_V = [
        -0.5 * 0.03915,
        -0.5 * 0.323549,
    ]

    assert V == expected_V

    np.testing.assert_allclose(
        gap[0],
        expected_d,
    )

    np.testing.assert_allclose(
        gap[1],
        expected_id,
        atol=1e-12,
    )


def test_d_plus_id_shear_preserves_nonzero_pairing():
    _, gap = generate_gap_parameters(
        gap_pairing="d_plus_id",
        strain="shear",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    assert np.any(gap[0] != 0)
    assert np.any(gap[1] != 0)


# ============================================================
# d + i g
# ============================================================

def test_d_plus_ig_perpendicular():

    V, gap = generate_gap_parameters(
        gap_pairing="d_plus_ig",
        strain="uniaxial",
        a1x=1.0,
        a1y=1.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected_d = np.array([
        [0, 2, 0],
        [-2, 0, -2],
        [0, 2, 0],
    ])

    expected_ig = np.zeros((3, 3))

    expected_V = [
        -0.5 * 0.03928,
        -0.5 * 0.496063,
    ]

    assert V == expected_V

    np.testing.assert_allclose(
        gap[0],
        expected_d,
    )

    np.testing.assert_allclose(
        gap[1],
        expected_ig,
        atol=1e-12,
    )


def test_d_plus_ig_shear():

    V, gap = generate_gap_parameters(
        gap_pairing="d_plus_ig",
        strain="shear",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected_d = np.array([
        [0, 2, 0],
        [-2, 0, -2],
        [0, 2, 0],
    ])

    expected_ig = np.zeros((3, 3))

    expected_V = [
        -0.5 * 0.03928,
        -0.5 * 0.496063,
    ]

    assert V == expected_V

    np.testing.assert_allclose(
        gap[0],
        expected_d,
    )

    np.testing.assert_allclose(
        gap[1],
        expected_ig,
        atol=1e-12,
    )


# ============================================================
# chiral p-wave
# ============================================================

def test_chiral_p_perpendicular():

    V, gap = generate_gap_parameters(
        gap_pairing="chiral_p",
        strain="uniaxial",
        a1x=1.0,
        a1y=1.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected_px = np.zeros((3, 3))
    expected_py = np.zeros((3, 3))

    expected_V = [
        -0.5 * 0.03915,
        -0.5 * 0.323549,
    ]

    assert V == expected_V

    np.testing.assert_allclose(
        gap[0],
        expected_px,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        gap[1],
        expected_py,
        atol=1e-12,
    )


def test_chiral_p_shear():

    V, gap = generate_gap_parameters(
        gap_pairing="chiral_p",
        strain="shear",
        a1x=1.0,
        a1y=0.0,
        nk=nk,
        kx=kx,
        ky=ky,
    )

    expected_px = np.zeros((3, 3))
    expected_py = np.zeros((3, 3))

    expected_V = [
        -0.5 * 0.03915,
        -0.5 * 0.323549,
    ]

    assert V == expected_V

    np.testing.assert_allclose(
        gap[0],
        expected_px,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        gap[1],
        expected_py,
        atol=1e-12,
    )
import numpy as np

from src.delta_sum import delta_sum


def test_delta_sum():
    # Test case 1: Simple case with known values
    C = 1.0
    Delta = 1.0
    fe = 0.5
    Ek = 4.0
    pairing = 1.0
    expected_output = 0
    assert delta_sum(C, Delta, fe, Ek, pairing) == expected_output

    # Test case 2: Zero gap function
    C = 1.0
    Delta = 0.0
    fe = 0.5
    Ek = 4.0
    pairing = 1.0
    expected_output = 0
    assert delta_sum(C, Delta, fe, Ek, pairing) == expected_output

    # Test case 3: Negative eigen energy
    C = 1.0
    Delta = 2.0
    fe = 0.5
    Ek = -4.0
    pairing = 1.0
    expected_output = C * (Delta * pairing * (1 - 2 * fe) / Ek)
    assert delta_sum(C, Delta, fe, Ek, pairing) == expected_output

    # Test case 4: s-wave pairing symmetry
    C = 1.0
    Delta = 1.0
    fe = 0.25
    Ek = 1.0
    pairing = 1.0  # s-wave pairing symmetry
    expected_output = 0.5
    assert delta_sum(C, Delta, fe, Ek, pairing) == expected_output

    # Test case 5 values in matrix form
    C = 1
    Delta = np.array([[1.0, 0.0], [0.0, 1.0]])
    fe = 0.25
    Ek = 0.5
    pairing = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert delta_sum(C, Delta, fe, Ek, pairing)  == 0
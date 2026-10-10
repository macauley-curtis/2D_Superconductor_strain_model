import numpy as np

from src.fermi_dirac import fermi_dirac_func


def test_fermi_dirac_func():
    # Test case 1: e = 0, kt = 1
    e = 0
    kt = 1
    expected_output = 0.5
    assert fermi_dirac_func(e, kt) == expected_output

    # Test case 2: e = 1, kt = 1
    e = 1
    kt = 1
    expected_output = 1 / (np.exp(1) + 1)
    assert fermi_dirac_func(e, kt) == expected_output

    # Test case 3: e = -1, kt = 1
    e = -1
    kt = 1
    expected_output = 1 / (np.exp(-1) + 1)
    assert fermi_dirac_func(e, kt) == expected_output

    # Test case for the high-temperature limit
    e = -100
    kt = np.inf
    expected_output = 0.5
    assert fermi_dirac_func(e, kt) == expected_output
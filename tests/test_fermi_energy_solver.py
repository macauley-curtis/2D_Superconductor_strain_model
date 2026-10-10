import numpy as np

from src.fermi_energy_solver import fermi_energy_solver


def test_fermi_energy_solver_():
    
    e = np.array([0.0, 0.0, 0.0])
    Ne = 4.0
    gamma_k = np.array([0.0, 0.0, 0.0])
    KbT = 1.0

    result = fermi_energy_solver(
        e,
        Ne,
        gamma_k,
        KbT,
    )

    expected = 0.5


    assert result == expected
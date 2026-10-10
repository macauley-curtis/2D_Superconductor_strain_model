import numpy as np

from src.calculate_tc import calculate_tc


def test_calculate_tc():
    temperature = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    Delta = np.array([
        [
            10.0,
            8.0,
            5.0,
            0.9,
            0.5,
        ]
    ])

    tolerance_ratio = 0.1

    expected = 3.0

    result = calculate_tc(
        Delta,
        temperature,
        tolerance_ratio,
        hopping_step=1,
    )

    np.testing.assert_equal(
        result,
        expected,
    )

def test_calculate_tc_prints_warning_when_tc_not_found(capsys):
    """Test that a warning is printed when Tc is not found."""

    Delta = np.array([
        [1.0, 0.8, 0.6],
        [1.0, 0.9, 0.7],
    ])

    temperature = np.array([0.1, 0.2, 0.3])

    result = calculate_tc(
        Delta=Delta,
        temperature=temperature,
        tolerance_ratio=0.1,
        hopping_step=2,
    )

    captured = capsys.readouterr()

    assert "Warning: Tc not found for hopping 0." in captured.out
    assert "Warning: Tc not found for hopping 1." in captured.out
import pytest

from src.return_n_components import return_gap_components


@pytest.mark.parametrize(
    "gap_pairing, expected",
    [
        ("s_wave", 1),
        ("d_wave", 1),
        ("ext_s_wave", 1),
        ("d_plus_id", 2),
        ("d_plus_ig", 2),
        ("chiral_p", 2),
    ],
)
def test_return_n_components(gap_pairing, expected):
    assert return_gap_components(gap_pairing) == expected
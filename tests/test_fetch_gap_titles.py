import pytest

from src.fetch_gap_titles import fetch_gap_titles


@pytest.mark.parametrize(
    "gap_pairing, expected",
    [
        (
            "s_wave",
            ("s", "one component"),
        ),
        (
            "d_wave",
            (
                "d$_{x^2-y^2}$",
                "one component",
            ),
        ),
        (
            "ext_s_wave",
            (
                "s$^{\\pm}$",
                "one component",
            ),
        ),
        (
            "d_plus_id",
            ("d+id", "two component", "$d_{x^2-y^2}$", "$d_{xy}$"),
        ),
        (
            "d_plus_ig",
            ("d+ig", "two component", "$d_{x^2-y^2}$", "g"),
        ),
        (
            "chiral_p",
            ("p$_x$ + ip$_y$", "two component", "p$_x$", "p$_y$"),
        ),
    ],
)
def test_gap_titles(gap_pairing, expected):
    assert fetch_gap_titles(gap_pairing) == expected

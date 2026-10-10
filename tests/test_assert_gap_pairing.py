import pytest

from src.utils.assert_gap_pairing import assert_gap_pairing


def test_assert_gap_pairing():
    # Test valid gap pairings
    valid_gap_pairings = [
        "s_wave",
        "d_wave",
        "ext_s_wave",
        "chiral_p",
        "d_plus_id",
        "d_plus_ig",
    ]

    for gap_pairing in valid_gap_pairings:
        assert_gap_pairing(gap_pairing)
        

    invalid_gap_pairings = [
        "invalid_gap_pairing",
        "",
        123,
        None,
        [],
        {},
    ]

    for gap_pairing in invalid_gap_pairings:
        with pytest.raises((TypeError, ValueError)):
            assert_gap_pairing(gap_pairing)
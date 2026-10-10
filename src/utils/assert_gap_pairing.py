def assert_gap_pairing(gap_pairing):
    """
    Assert that the given gap pairing is valid.

    Parameters:
    gap_pairing (str): The gap_pairing to validate.

    Raises:
    TypeError: If the gap_pairing is not valid.
    """
    if not isinstance(gap_pairing, str):
        raise TypeError("gap_pairing must be a string.")

    if not gap_pairing:
        raise ValueError("gap_pairing cannot be empty.")

    valid_gap_pairings = (
        "s_wave",
        "d_wave",
        "ext_s_wave",
        "chiral_p",
        "d_plus_id",
        "d_plus_ig",
    )

    if gap_pairing not in valid_gap_pairings:
        raise ValueError(
            f"Invalid gap_pairing: {gap_pairing}. Must be one of "
            "'s_wave', 'd_wave', 'ext_s_wave', 'chiral_p', 'd_plus_id' or 'd+ig'."
        )
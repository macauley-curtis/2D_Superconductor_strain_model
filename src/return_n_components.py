def return_gap_components(gap_pairing):
    """Return the number of gap-function components."""
    gap_components = {
        "s_wave": 1,
        "d_wave": 1,
        "ext_s_wave": 1,
        "d_plus_id": 2,
        "d_plus_ig": 2,
        "chiral_p": 2,
    }

    return gap_components[gap_pairing]
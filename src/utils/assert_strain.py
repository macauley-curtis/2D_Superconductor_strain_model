def assert_strain(strain):
    """
    Assert that the given strain is valid.

    Parameters:
    strain (str): The strain to validate.

    Raises:
    TypeError: If the strain is not valid.
    """
    if not isinstance(strain, str):
        raise TypeError("Strain must be a string.")

    if not strain:
        raise ValueError("Strain cannot be empty.")

    valid_strains = ("uniaxial", "c_axis", "shear")

    if strain not in valid_strains:
        raise ValueError(
            f"Invalid strain: {strain}. Must be one of "
            "'uniaxial', 'c_axis', or 'shear'."
        )
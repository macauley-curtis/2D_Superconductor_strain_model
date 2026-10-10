import pytest

from src.utils.assert_strain import assert_strain


@pytest.mark.parametrize(
    "strain",
    [
        "uniaxial",
        "c_axis",
        "shear",
    ],
)
def test_valid_strain(strain):
    assert_strain(strain)


@pytest.mark.parametrize(
    "strain",
    [
        1,
        1.0,
        None,
        True,
        False,
        ["uniaxial"],
        ["c_axis", "shear"],
        ("uniaxial",),
        ("c_axis", "shear"),
    ],
)
def test_strain_must_be_string(strain):
    with pytest.raises(TypeError, match="Strain must be a string."):
        assert_strain(strain)


def test_strain_cannot_be_empty():
    with pytest.raises(ValueError, match="Strain cannot be empty."):
        assert_strain("")


@pytest.mark.parametrize(
    "strain",
    [
        "uniax",
        "uniaxialx",
        "caxis",
        "shearx",
        "UNIXIAL",
        "invalid",
    ],
)
def test_invalid_strain(strain):
    with pytest.raises(ValueError, match="Invalid strain:"):
        assert_strain(strain)

import pytest

from src.utils.classify_strain_type import classify_strain_type


def test_classify_strain_type():
    assert classify_strain_type("uniaxial") == "perpendicular"
    assert classify_strain_type("c_axis") == "perpendicular"
    assert classify_strain_type("shear") == "shear"
    
    with pytest.raises(ValueError):
        classify_strain_type("foo") 
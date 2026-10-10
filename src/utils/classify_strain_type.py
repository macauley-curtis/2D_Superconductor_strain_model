from src.utils.assert_strain import assert_strain
def classify_strain_type(strain):
    
    """
    A helper function to classify strain type for gap parameter generation.
    :param strain: type of strain; uniaxial, c_axis or shear
    :return: returns the strain type as either "perpendicular" or "shear
    """
    
    assert_strain(strain)
    
    
    if strain in ["uniaxial", "c_axis"]:
        return "perpendicular"
    else:
        return "shear"
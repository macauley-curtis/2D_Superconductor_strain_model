from src.round_femi_vals import round_fermi_vals


def test_round_fermi_vals():

    assert round_fermi_vals("Uniaxial", 
                     [1, 2, 4],
                     [0, 1, 2],
                     shear_t_unit=0,
                     perpendicular_t_unit=2) == (0.5, 1.0, 2.0)

    assert round_fermi_vals("shear",
                        [1, 2, 4],
                        [0, 1, 2],
                        shear_t_unit=1,
                        perpendicular_t_unit=0) == (1.0, 2.0, 4.0)
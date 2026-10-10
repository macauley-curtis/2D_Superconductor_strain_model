
def round_fermi_vals(
        strain, 
        hopping_array, 
        fermi_vals, 
        shear_t_unit=36.73 * 10**-3,
        perpendicular_t_unit=81.62 * 10**-3):
    """
    param strain: string specifying the strain type
    param hopping_array: array of hopping values
    param fermi_vals: array of fermi values
    param shear_t_unit: shear hopping unit value
    param perpendicular_t_unit: perpendicular hopping unit value
    return: rounded hopping values at fermi levels

    """
    if strain == "shear":
        t_unit = shear_t_unit
    else:
        t_unit = perpendicular_t_unit

    val_1, val_2, val_3 = (
        round(hopping_array[fermi_vals[0]] / t_unit, 2),
        round(hopping_array[fermi_vals[1]] / t_unit, 2),
        round(hopping_array[fermi_vals[2]] / t_unit, 2),
    )
    return val_1, val_2, val_3
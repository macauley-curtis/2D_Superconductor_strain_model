def delta(de, x):
    """
    :param de: screening energy - used for numerical approximation
    :param x: posisitonal argument, for the dirac function centred on x
    :return: delta dirac approximation
    """
    return de / ((x**2) + (de**2))

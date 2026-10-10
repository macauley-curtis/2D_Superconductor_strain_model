import numpy as np

from src.utils import assert_gap_pairing, assert_strain, classify_strain_type


def generate_gap_parameters(
    gap_pairing, 
    strain, 
    a1x, 
    a1y, 
    nk,
    kx,
    ky):
    
    """
    :param gap_pairing: type of pairing; s_wave, d_wave, ext_s_wave, chiral_p, d_plus_id, d_plus_ig
    :param strain: type of strain; uniaxial, c_axis or shear
    :param a1x: x component of lattice coordinate a1
    :param a1y: y component of lattice coordinate a1
    :param nk: number of k points, such that k-space grid will be nk x nk
    :param kx: kx values for the k-space grid
    :param ky: ky values for the k-space grid
    
    :return: Interaction potential with multiplicity constant - this is 
    numerically calculated by matching to experimental data, 
    gap pairing, titles for plots  specific to pairing/strain combination
    
    """
    
    # Validate the strain input
    assert_strain.assert_strain(strain)
    assert_gap_pairing.assert_gap_pairing(gap_pairing)
    
    strain_type = classify_strain_type.classify_strain_type(strain)

    gap_parameters = {
        "s_wave": {
            "V": -0.035,
            "strains": {
                "perpendicular": {
                    "gap_pairing": 1.0 * np.ones((nk, nk)),
                },
                "shear": {
                    "gap_pairing": 1.0 * np.ones((nk, nk)),
                },
            },
        },

        "d_wave": {
            "V": -0.0394 * 0.5,
            "strains": {
                "perpendicular": {
                    "gap_pairing": np.cos(kx) - np.cos(ky),
                },
                "shear": {
                    "gap_pairing": (
                        np.cos(a1x * kx + a1y * ky)
                        - np.cos(a1y * kx + a1x * ky)
                    ),
                },
            },
        },

        "ext_s_wave": {
            "V": -0.06483,
            "strains": {
                "perpendicular": {
                    "gap_pairing": np.cos(kx) + np.cos(ky),
                },
                "shear": {
                    "gap_pairing": (
                        np.cos(a1x * kx + a1y * ky)
                        + np.cos(a1y * kx + a1x * ky)
                    ),
                },
            },
        },

        "d_plus_id": {
            "V": [
                -0.5 * 0.03915,
                -0.5 * 0.323549,
            ],
            "strains": {
                "perpendicular": {
                    "gap_pairing": [
                        np.cos(kx) - np.cos(ky),
                        np.sin(kx) * np.sin(ky),
                    ],
                },
                "shear": {
                    "gap_pairing": [
                        (
                            np.cos(a1x * kx + a1y * ky)
                            - np.cos(a1y * kx + a1x * ky)
                        ),
                        (
                            np.sin(a1x * kx + a1y * ky)
                            * np.sin(a1y * kx + a1x * ky)
                        ),
                    ],
                },
            },
        },

        "d_plus_ig": {
            "V": [
                -0.5 * 0.03928,
                -0.5 * 0.496063,
            ],
            "strains": {
                "perpendicular": {
                    "gap_pairing": [
                        np.cos(kx) - np.cos(ky),
                        np.sin(kx)
                        * np.sin(ky)
                        * (np.cos(kx) - np.cos(ky)),
                    ],
                },
                "shear": {
                    "gap_pairing": [
                        (
                            np.cos(a1x * kx + a1y * ky)
                            - np.cos(a1y * kx + a1x * ky)
                        ),
                        (
                            np.sin(a1x * kx + a1y * ky)
                            * np.sin(a1y * kx + a1x * ky)
                            * (
                                np.cos(a1x * kx + a1y * ky)
                                - np.cos(a1y * kx + a1x * ky)
                            )
                        ),
                    ],
                },
            },
        },

        "chiral_p": {
            "V": [
                -0.5 * 0.03915,
                -0.5 * 0.323549,
            ],
            "strains": {
                "perpendicular": {
                    "gap_pairing": [
                        np.sin(kx),
                        np.sin(ky),
                    ],
                },
                "shear": {
                    "gap_pairing": [
                        np.sin(a1x * kx + a1y * ky),
                        np.sin(a1y * kx + a1x * ky),
                    ],
                },
            },
        },
    }
    parameters = gap_parameters[gap_pairing]
    V = parameters["V"]
    gap_pairing = parameters["strains"][strain_type]["gap_pairing"]
    
    return V, gap_pairing


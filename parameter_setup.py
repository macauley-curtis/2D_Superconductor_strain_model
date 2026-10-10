# =========================================================================== #
#                           CALCULATION TYPE                                  #
# =========================================================================== #
strain = "strain"
gap_pairing = "gap_pairing"


# =========================================================================== #
#                           CALCULATION DETAILS                               #
# =========================================================================== #


# ---------------------------------------------------------------------------- #
# Available modes
# ---------------------------------------------------------------------------- #
#
# "calculate" -> Calculate the quantity
# "load"      -> Load a previously calculated quantity
# "skip"      -> Skip the calculation and do not load
# ---------------------------------------------------------------------------- #


# =========================================================================== #
#                           CALCULATION MODES                                 #
# =========================================================================== #
# Hopping array
hopping_array_mode = "calculate"
energyband_mode = "calculate"
dos_mode = "calculate"
delta_mode = "calculate"
tc_mode = "calculate"


# ============================================================================ #
#                           SAVE / LOAD SETTINGS                              #
# ============================================================================ #
save_hopping_array = True
save_energyband = True
dos_mode = "calculate"
save_delta = True
save_tc = True


# Energy band

energyband_name = "energy_band.npy"
energyband_plot = True

# Superconducting gap / Delta


delta_name = "delta_temperature.npy"
delta_plot = True

# Critical temperature

tc_name = "critical_temperature.npy"
tc_plot = True

# Density of states

dos_name = "density_of_states.npy"
dos_plot = True


# ---------------------------------------------------------------------------- #
#                           PARAMETER SETUP                                    #
# ---------------------------------------------------------------------------- #

# Save a copy of the parameter configuration used for this calculation.
save_parameters = True



# Lattice params
a1x = 1.0
a1y = 0.0

# Set up k-space
nk = 500
BZ_edge = 1.0

# Hopping params
tx = 81.62e-3
tprime = 36.73e-3
e0 = -131e-3

hopping_step = 10
t1 = 1.0
t2 = 1.14

# Temperature params
temperature_max = 3.5
temperature_step = 200

# density of states params
dos_bound = 0.5
dos_step = 1000
screening_energy = 0.01

# computation params
convergence_num = 2000
conv_tolerance = 1.8e-8
delta_guess = 0.01
mixing = 0.5

tolerance_ratio = 0.1

# Physical constants
Kb = (1.38e-23) / (1.6e-19) 

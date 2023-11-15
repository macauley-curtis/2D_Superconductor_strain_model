# 2D_Superconductor_strain_model
The script developed here were created for my personal PhD project for exploring the effect of strain on the superconducting critical temperature

Strain is modelled via changing the hopping integrals, in increase in t_x is a decrease in the physical bond length in the x direction of the crystal lattice.



The energyband is used in all following caluclations, the hopping integral is inherently involved in this calculation but saved for when plotting as a function of hopping anisotropy is used. If desired, the Fermi energy can be calculated by subtracted the first value of epsilon_0 form the whole array s.t fermi_energy = epsilon_0[:] - epsilon_0[0] as the Fermi energy is set to zero.

The Delta calculation will calculate the gap function as a fucntion of temperature and hopping anisotropy, returning the gap function and also the critcal temperature. 

Dos is the denisty of states, it calculates the desity of states in the normal state (representing s-wave BCS superconductors), and then weighted dos for all other models which is the normal state weighted by the gap symmetry Gamma(k).
This provides insight into the superconductting characteristics. 



You can specify the paramaters conercned to the scripts as stated, you can speificy the type of gap pairing (gap_pairing="") and the strain (strain="") which automatically pulls the correct previous data, and writes the correct output. It is all the same format so they all work together. The possible values are specified in the function "gap_symmetries" ane the strain is either: "uniaxial", "shear" or "c_axis".

In main, there is opperutnity to alter both some of the physical parameters (for example physical values specific to the material) and also the more computational values linked to convergence, array size etc to optimize run-time and computational loads on your specific machines. I can run on Mac OS, Macbook pro M1 for s-wave, uniaixal, 1:1.14 in 20 steps on a 700x700 grid in about 6 minutes.

For further details on how to read the outputs in context, this paper authored by me and my supervisor outlines all details of the resulting plots and what they mean for the case study of Sr2RuO4 - https://arxiv.org/abs/2209.00300.



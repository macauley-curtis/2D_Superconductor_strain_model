# 2D_Superconductor_strain_model
The scripts developed here were created for my personal PhD project for exploring the effect of strain on the superconducting critical temperature

Strain is modelled via changing the hopping integrals, in increase in t_x is a decrease in the physical bond length in the x direction of the crystal lattice.

The order of running is as follows:

1) Energyband 
2) Delta Calculation OR Dos (depending on what you wish to calculate) 
3) plot


Nothing will run without first calculating the energyband which outputs the hopping parameters, the energyband and the onsite energy epsilon_0. 
The energyband is used in all following caluclations, the hopping integral is inherently involved in this calculation but saved for when plotting as a function of hopping anisotropy is used. If desired, the Fermi energy can be calculated by subtracted the first value of epsilon_0 form the whole array s.t fermi_energy = epsilon_0[:] - epsilon_0[0] as the Fermi energy is set to zero.

The Delta calculation will calculate the gap function as a fucntion of temperature and hopping anisotropy, returning the gap function and also the critcal temperature. 

Dos is the denisty of states, it calculates the desity of states in the normal state (representing s-wave BCS superconductors), and then weighted dos for all other models which is the normal state weighted by the gap symmetry Gamma(k).
This provides insight into the superconductting characteristics. 

And plot... plots depending on what you wish to plot.

For each script, you can specify the paramaters conercned to the scripts as stated, however in each you can speificy the type of gap pairing (gap_pairing="") and the strain (strain="") which automatically pulls the correct previous data, and writes the correct output. It is all the same format so they all work together. The possible values are specified in the function "gap_symmetries" ane the strain is either: "uniaxial", "shear" or "c_axis".


The Delta script will throw up some runtime and exponential errors, this is due ot the handling of incriedbly small numbers in the femri-driac function, nummpy takes care of this by setting this to zero when needed and this is what it is warning you it is doing. It is a o k 


For further details on how to read the outputs in context, this paper authored by me and my supervisor outlines all details of the resulting plots and what they mean for the case study of Sr2RuO4 - https://arxiv.org/abs/2209.00300.



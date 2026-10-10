# 2D Superconductor Strain Model

A Python model for investigating the effect of lattice strain on superconductivity in a two-dimensional tight-binding system. This code was developed as part of my PhD research to explore how strain-induced changes in electronic structure influence the superconducting gap, critical temperature, and density of states.

The model supports several strain configurations and superconducting pairing symmetries, allowing their effects to be investigated within a consistent computational framework.

Scientific Background
Strain is modeled through changes in the electronic hopping integrals, which describe the transfer of electrons between lattice sites. For example, an increase in the nearest-neighbour hopping parameter $t_x$ corresponds to a decrease in the physical bond length along the
$x$-direction, within the model's assumed relationship between hopping and lattice spacing.

The resulting changes in the electronic band structure affect the Fermi surface and density of states, which in turn influence the superconducting gap and critical temperature.

The model consists of three main calculations:

## 1. Energy Band Structure
The energy-band calculation determines the electronic dispersion $\varepsilon_k$ throughout the Brillouin zone for a range of strain-induced hopping parameters.

The chemical potential is adjusted at each strain value to maintain a constant electron number. The resulting energy bands are saved and reused by subsequent calculations, avoiding unnecessary recomputation.

The hopping parameters are also saved separately, allowing calculated properties to be plotted against hopping anisotropy or the relevant normalised hopping parameter.

The energy reference is set by the chemical-potential convention used in the calculation. Where required, a relative chemical-potential shift can be obtained by subtracting its unstrained value:

`fermi_energy = epsilon_0[:] - epsilon_0[0]`
## 2.  Superconducting Gap and Critical Temperature
The superconducting gap calculation solves the self-consistent gap equation as a function of temperature and strain.

For each hopping configuration, the calculation returns the temperature-dependent gap and an estimate of the superconducting critical temperature
$T_c$.

Both single-component and two-component order parameters are supported. The two-component models allow complex order parameters, enabling the investigation of pairing states such as chiral or mixed-symmetry superconducting phases.

The resulting data can be used to investigate how strain affects the superconducting gap and critical temperature for different pairing symmetries.

A general attratice interaction potential is assumed and calculatred by matching
the unstrained Gap parameters to experimental values, tuning $T_c$.

## 3. Density of States (DOS)
The density-of-states calculation evaluates the normal-state DOS and the corresponding gap-symmetry-weighted DOS.

The normal-state DOS characterises the available electronic states as a function of energy. The weighted DOS incorporates the momentum dependence of the selected pairing symmetry,
$\Gamma(k)$ providing additional insight into how the electronic structure relates to the candidate superconducting order parameter.

These quantities can be used alongside the energy-band and gap calculations to investigate the relationship between strain, electronic structure, and superconducting properties.

### Supported Strain Configurations
The model supports three strain configurations:

| Strain | Effective change |
| -------| -----------------|
|uniaxial | changes the nearest neighbour hopping parameters along the target axis|
|c_axis  |changes the in-plane hopping parameters according to the model's c-axis strain|
|shear | primarily modifies the next-nearest neighbour hopping parameters and lattice geometry to represent shear strain|

Each configuration has its own hopping parameters and associated energy-band data.

### Supported Pairing Symmetries
The following pairing symmetries are available through the gap_pairing argument:

Argument Pairing symmetry
|pairing name | pairing symbol |
|------------ | -------------- |
| `s_wave`    | $s-wave$ |
| `ext_s_wave` | ($s^{\pm}$) |
| `d_wave` | $d_{x^2−y^2}$ |
| `d_plus_id` |$d_{x^2−y^2} + id_{xy}$ |
| `d_plus_ig` | $d_{x^2−y^2} +ig$ |
| `chiral_p` | $p_x + ip_y$ |

## Getting Started

### Requirements
The model uses Python and the following scientific libraries:

`NumPy`
`Matplotlib`
`SciPy`
Install the dependencies using the repository's requirements file, if available:

`python -m pip install -r requirements.txt`

# Setup file

`parameter_setup.py` contains all of the parameters required for the calculation.
It's split up by first choosing the gap pairing and strain combination to claculate.
Then deciding whether to calculate, load and save each segmant of the calculation.

And finally the physical parameters. The file pushed to github contains roughly the correct
parameters used during calculation, there has since been some changes to how the interation
potential is treated so may need adjusting the unstrained version to get $T_c = 0$ for
each pairing.

### Project Structure
The repository is organised around the following directories and scripts:

`src/` — source code for the model and its calculation routines.
`tests/` — tests for the model's functions and behaviour.
`results/` — generated numerical data, plots, and other calculation outputs -
`helpers.py` contains functions for working out the save paths, creating and
saving/loading to/from these paths.
`parameter_setup.py` — configuration of model parameters.
`run_calculations.py` — entry point for running calculations.

When adapting the code to another material, ensure that the physical parameters and their units are consistent throughout the calculation.

The model contains a fermi-sruface style screening only calculating energies within
`cutoff_energy` of the femri surface, vastly reducing the k-space gird points needed to caluclate - however overdoing this also increases volatility in the self consistent calculation. So
a compromise needs to be made between `nk` and `cutoff_energy`.

Actual runtimes will vary with the machine, calculation settings, and enabled outputs.

An example run on a Macbook M4 pro for `stain = uniaxial` and
`gap_pairing = "d-wave"`  with `nk = 500`, `temperature_step = 300` and `hopping_step = 10` is as below:

```python
 ek_run_time: 0.11 seconds
 dos_run_time: 3.13 seconds
 delta_run_time: 29.33 seconds
 tc_run_time: 0.00 seconds
```

`utils.helpers.save_params_setup()` will save the parameters and runtimes into the target folder when `save_parameters = True`.

When the save for either the data or plots are set to `True` they will be saved alongside the parameters.
You may then set the modes to `load` to pull these rather than recalculating.

## Research and Further Reading

This code was developed for my PhD research into the effects of strain on superconductivity, with particular application to strontium ruthenate ($Sr_2RuO_4$).

The following publications provide further context for the physical model, interpretation of the calculated quantities, and scientific motivation.

Research paper: Strain effects in $Sr_2RuO_4$ — [arXiv:2209.00300](https://arxiv.org/abs/2209.00300)
PhD thesis: [Mac Curtis — PhD thesis, University of Bristol](https://research-information.bris.ac.uk/ws/portalfiles/portal/405539900/Mac_Curtis_Thesis_1_.pdf)
Please refer to these works for the detailed theoretical background and interpretation of the results.

## Acknowledgements
This project was developed as part of my personal PhD research. The code is intended to support further exploration of strain-dependent electronic structure and superconductivity in two-dimensional lattice models.

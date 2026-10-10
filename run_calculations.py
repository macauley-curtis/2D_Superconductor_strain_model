import time

import numpy as np

import parameter_setup as params
from src import (
    calculate_delta,
    calculate_tc,
    create_energyband,
    create_hopping_range,
    create_square_k_space,
    density_of_states,
    generate_gap_params,
    return_n_components,
)
from src.utils import (
    assert_gap_pairing,
    assert_strain,
)
from src.utils.helpers import (
    create_results_directories,
    generate_data_path,
    load_data,
    runtime_summary,
    save_data,
    save_parameters_setup,
)


def main():
    
    # Need documentation 
    # Need tests // coverage?
    # Refactors
    # Look at order of arguments for consitency


    #===========================================================================
    # Validate strain and gap pairing
    #===========================================================================    
    
    assert_strain.assert_strain(params.strain)
    assert_gap_pairing.assert_gap_pairing(params.gap_pairing)

    #===========================================================================
    # Create results directories and generate file paths for saving into
    #===========================================================================
    
    create_results_directories(params.strain, params.gap_pairing)
    
    energy_band_name = generate_data_path(
        params.strain, 
        params.gap_pairing, 
        "energy_band"
    )
    dos_name = generate_data_path(
        params.strain, 
        params.gap_pairing, 
        "density_of_states"
    )
    delta_name = generate_data_path(
        params.strain, 
        params.gap_pairing, 
        "delta"
    )
    tc_name = generate_data_path(
        params.strain, 
        params.gap_pairing, 
        "tc"
    )
    hopping_array_name = generate_data_path(
        params.strain, 
        params.gap_pairing, 
        "hopping_array"
    )
    setup_params_name = generate_data_path(
        params.strain, 
        params.gap_pairing, 
        "parameter_setup",
        extension="txt"
    )
    
    #===============================================================================
    #  Set up physical arrays
    #===============================================================================
    
    kx,ky = create_square_k_space.create_square_k_space(
        params.nk, params.BZ_edge
        )
    
    temperature = np.linspace(
        0, params.temperature_max, params.temperature_step
        )
    
    epsilon = np.linspace(-params.dos_bound, params.dos_bound, params.dos_step)
    
    # Setup hopping array
    
    if params.hopping_array_mode == "calculate":
        hopping_array = create_hopping_range.create_hopping_range(
            strain=params.strain,
            t_x=params.tx,
            t_prime=params.tprime,
            t_1=params.t1,
            t_2=params.t2,
            hopping_step=params.hopping_step,
        )
    if params.hopping_array_mode == "load":
        hopping_array = load_data(hopping_array_name)

    
    # number of gap components 
    
    n_comps = return_n_components.return_gap_components(params.gap_pairing)
        
    # set up k-space gap-function
    
    if params.delta_mode == "calculate" or params.dos_mode == "calculate":
    
        V, gap_function = generate_gap_params.generate_gap_parameters(
            strain=params.strain,
            gap_pairing=params.gap_pairing,
            nk=params.nk,
            kx=kx,
            ky=ky,
            a1x=params.a1x,
            a1y=params.a1y
        )
        
    #===========================================================================
    # Calculate energyband 
    #===========================================================================
    ek_time_start = time.perf_counter()
    
    if params.energyband_mode == "calculate":    
        energy_band = create_energyband.create_energyband(
            strain= params.strain,
            n_k = params.nk,
            kx = kx,
            ky = ky,
            t_x = params.tx,
            t_prime = params.tprime,
            e_0 = params.e0,
            hopping_step = params.hopping_step,
            hopping_array = hopping_array
        )
        
    ek_run_time = time.perf_counter() - ek_time_start
    
    if params.energyband_mode == "load":
        energy_band = load_data(energy_band_name)
    
    #=========================================================================== #
    #  Calculate density of states
    #=========================================================================== #
    
    dos_time_start = time.perf_counter()
    
    if params.dos_mode == "calculate":
            
        dos = density_of_states.dos(
            gap_pairing=gap_function,
            e_k=energy_band,
            epsilon=epsilon,
            dos_step= params.dos_step,
            screening_energy=params.screening_energy,
            hopping_step=params.hopping_step,
            nk=params.nk,
            n_components=n_comps
        )
    dos_run_time = time.perf_counter() - dos_time_start
        
    if params.dos_mode == "load":
        dos = load_data(dos_name)
    
    
    #===========================================================================    
    #  Calculate delta 
    #===========================================================================
    delta_time_start = time.perf_counter()
    
    if params.delta_mode == "calculate":
        
        delta = calculate_delta.calculate_delta(
            strain=params.strain,
            e_k=energy_band,
            n_k=params.nk,
            convergence_num=params.convergence_num,
            hopping_step=params.hopping_step,
            temperature=temperature,
            temperature_step=params.temperature_step,
            conv_tolerance=params.conv_tolerance,
            mixing=params.mixing,
            delta_guess=params.delta_guess,
            V=V,
            gap_function=gap_function,
            Kb=params.Kb,
            energy_cutoff = 0.05,
            n_components=n_comps
        )
    
    delta_run_time = time.perf_counter() - delta_time_start
        
    if params.delta_mode == "load":
        delta = load_data(delta_name)
        
    #===========================================================================
    # Calculate Tc 
    #===========================================================================
    
    tc_time_start = time.perf_counter()
    if params.tc_mode == "calculate":
        if n_comps == 1:
            tc = calculate_tc.calculate_tc(
                Delta=delta[:, :, 0],
                temperature=temperature,
                tolerance_ratio=params.tolerance_ratio,
                hopping_step=params.hopping_step
            )
        else:
            tc1 = calculate_tc.calculate_tc(
                Delta=delta[:, :, 0],
                temperature=temperature,
                tolerance_ratio=params.tolerance_ratio,
                hopping_step=params.hopping_step
            )
            tc2 = calculate_tc.calculate_tc(
                Delta=delta[:, :, 1],
                temperature=temperature,
                tolerance_ratio=params.tolerance_ratio,
                hopping_step=params.hopping_step
            )
            tc = np.stack((tc1, tc2), axis=2)
            
    tc_end_time = time.perf_counter()
    tc_runtime = tc_end_time - tc_time_start
    
    
    if params.tc_mode == "load":
        tc = load_data(tc_name)
        
    #============================================================================
    # saving outputs
    #============================================================================
    
    
    runtime_summary([
        ("energyband", ek_run_time),
        ("dos", dos_run_time),
        ("delta", delta_run_time),
        ("tc", tc_runtime),
    ])  
    
    if params.save_hopping_array:
            save_data(hopping_array, hopping_array_name)
            
    if params.save_energyband:
            save_data(energy_band, energy_band_name)
            
    if params.save_dos:
            save_data(dos, dos_name)
            
    if params.save_parameters:
            save_parameters_setup(
                setup_params_name,
                params,
                ek_run_time=ek_run_time,
                dos_run_time=dos_run_time,
                delta_run_time=delta_run_time,
                tc_run_time=tc_runtime
            )
    if params.save_delta:
            save_data(delta, delta_name)
        
        
    if params.save_tc:
            save_data(tc, tc_name)
            
    
    if params.energyband_plot:
        from src.plot_results import plot_energy_band
        plot_energy_band(
            params=params,
            energy_band=energy_band,
            hopping_array=hopping_array,
            kx=kx,
            ky=ky
        )
        
    if params.tc_plot:
        from src.plot_results import plot_tc
        plot_tc(
            tc=tc,
            hopping_array=hopping_array,
            params=params
        )
    if params.delta_plot:
        from src.plot_results import plot_delta
        plot_delta(
            params=params,
            delta=delta,
            temperature=temperature,
            hopping_array=hopping_array,
            kx=kx,
            ky=ky
        )

    if params.dos_plot:
        from src.plot_results import plot_density_of_states
        plot_density_of_states(
            params=params,
            dos=dos,
            epsilon=epsilon,
            hopping_array=hopping_array,
            n_components=n_comps
        )

if __name__ == "__main__":
    main()

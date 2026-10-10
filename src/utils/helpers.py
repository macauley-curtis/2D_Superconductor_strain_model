import os
from pathlib import Path

import numpy as np

from src.utils.cli_output import toast


def save_data(data, filename):
    
    """
    
    Save data to a file, with overwrite protection.
    
    param data: Data to be saved (numpy array or other serializable format).
    param filename: Path to the file where data will be saved.
    
    """
    

    if os.path.exists(filename):

        toast(
            f"File already exists:\n{filename}",
            color="orange1",
            title="OVERWRITE WARNING",
        )

        response = input(
            "Overwrite existing file? [y/n]: "
        ).strip().lower()

        if response not in ("y", "yes"):
            toast(
                f"Keeping existing file: {filename}",
                color="yellow",
                title="Save Cancelled",
            )
            return

    np.save(filename, data)

    toast(
        f"Data saved to: {filename}",
        color="green",
        title="Save Complete",
    )



def load_data(filename):
    
    """
    Load data from a file, with existence check.
    
    param filename: Path to the file from which data will be loaded.
    return: Loaded data (numpy array or other format).
    
    """

    if not os.path.exists(filename):

        toast(
            f"File does not exist: {filename}",
            color="red",
            title="Load Failed",
        )

        raise FileNotFoundError(
            f"Cannot load '{filename}' because the file does not exist."
        )

    toast(
        f"Loading data from: {filename}",
        color="cyan",
        title="Loading",
    )

    return np.load(
        filename,
        allow_pickle=True,
    )

def create_results_directories(strain, gap_pairing):
    """
    Create the necessary directories for saving results 
    
    :param strain: Type of strain (e.g., 'uniaxial', 'c_axis', 'shear').
    :param gap_pairing: Type of gap pairing (e.g., 's_wave', 'd_wave').
    """

    dir = Path(f"results/{strain}/{gap_pairing}")

    dir.mkdir(parents=True, exist_ok=True)

    toast(
        f"Results directories created:{dir}",
        color="green",
        title="Directories Created",
    )
    
def generate_data_path(strain, gap_pairing, data_type, extension="npy"):
    """
    Generate a file path based on strain, gap pairing, and data type.

    :param strain: Type of strain (e.g., 'uniaxial', 'c_axis', 'shear').
    :param gap_pairing: Type of gap pairing (e.g., 's_wave', 'd_wave').
    :param data_type: Type of data (e.g., 'hopping_array', 'energy_band').
    
    """
    
    file_name = f"{strain}_{gap_pairing}_{data_type}.{extension}"
    file_path = os.path.join(f"results/{strain}/{gap_pairing}", file_name)
    
    return file_path

def generate_plot_path(strain, gap_pairing, data_type, extension="mp4"):
    """
    Generate a file path for plots based on strain, gap pairing, and data type.

    :param strain: Type of strain (e.g., 'uniaxial', 'c_axis', 'shear').
    :param gap_pairing: Type of gap pairing (e.g., 's_wave', 'd_wave').
    :param data_type: Type of data (e.g., 'hopping_array', 'energy_band').

    """
    
    if data_type == "tc":
        file_name = f"{strain}_{gap_pairing}_{data_type}.png"
        file_path = os.path.join(
            f"results/{strain}/{gap_pairing}", file_name
            )
    else:
        file_name = f"{strain}_{gap_pairing}_{data_type}.{extension}"
        file_path = os.path.join(
            f"results/{strain}/{gap_pairing}", file_name
            )
    return file_path


def save_parameters_setup(filepath, 
                          params, 
                          ek_run_time, 
                          dos_run_time,
                          delta_run_time, 
                          tc_run_time
                          ):
    """
    Save selected calculation parameters to a text file.
    
    :param filepath: Path to the file where parameters will be saved.
    :param params: Parameter object containing calculation settings.
    :param ek_run_time: Runtime for energy band calculation.
    :param dos_run_time: Runtime for density of states calculation.
    :param delta_run_time: Runtime for superconducting gap calculation.
    :param tc_run_time: Runtime for critical temperature calculation.
    
    """

    parameter_text = f"""\
    strain: {params.strain}
    gap_pairing: {params.gap_pairing}

    a1x: {params.a1x}
    a1y: {params.a1y}

    nk: {params.nk}
    BZ_edge: {params.BZ_edge}

    tx: {params.tx:.5e}
    tprime: {params.tprime:.5e}
    e0: {params.e0:.5e}

    hopping_step: {params.hopping_step}
    t1: {params.t1}
    t2: {params.t2}

    temperature_max: {params.temperature_max}
    temperature_step: {params.temperature_step}

    convergence_num: {params.convergence_num}
    conv_tolerance: {params.conv_tolerance:.5e}
    delta_guess: {params.delta_guess}
    mixing: {params.mixing}
    tolerance_ratio: {params.tolerance_ratio}

    Kb: {params.Kb:.5e}
    
    ek_run_time: {ek_run_time:.2f} seconds
    dos_run_time: {dos_run_time:.2f} seconds
    delta_run_time: {delta_run_time:.2f} seconds
    tc_run_time: {tc_run_time:.2f} seconds
    """

    with open(filepath, "w") as f:
        f.write(parameter_text)

    toast(
        f"Parameters saved to: {filepath}",
        color="green",
        title="Parameters Saved",
    )


def runtime_summary(runtimes):
    
    """
    Print a summary of the runtimes for each calculation step.
    """

    for name, runtime in runtimes:

        if runtime == 0.0:
            toast(
                f"{name} not calculated.",
                color="cyan",
                title="Warning",
            )
        else:
            toast(
                f"{name}: {runtime:.2f} seconds",
                color="green",
                title="Runtime Summary",
            )
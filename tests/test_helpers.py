from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch
import numpy as np
import pytest

from src.utils import helpers


# ============================================================
# save_data
# ============================================================

def test_save_data_new_file(tmp_path):
    """Test saving data when the file does not already exist."""

    filename = tmp_path / "data.npy"
    data = np.array([1.0, 2.0, 3.0])

    with patch("src.utils.helpers.toast") as mock_toast:
        helpers.save_data(data, str(filename))

    assert filename.exists()
    np.testing.assert_array_equal(np.load(filename), data)

    mock_toast.assert_called_once_with(
        f"Data saved to: {filename}",
        color="green",
        title="Save Complete",
    )


def test_save_data_overwrite_confirmed(tmp_path, monkeypatch):
    """Test overwriting an existing file when the user confirms."""

    filename = tmp_path / "data.npy"

    np.save(filename, np.array([1, 2]))
    new_data = np.array([3, 4])

    monkeypatch.setattr("builtins.input", lambda _: "yes")

    with patch("src.utils.helpers.toast") as mock_toast:
        helpers.save_data(new_data, str(filename))

    np.testing.assert_array_equal(np.load(filename), new_data)

    assert mock_toast.call_args_list == [
        call(
            f"File already exists:\n{filename}",
            color="orange1",
            title="OVERWRITE WARNING",
        ),
        call(
            f"Data saved to: {filename}",
            color="green",
            title="Save Complete",
        ),
    ]


@pytest.mark.parametrize("response", ["n", "no", "", "anything"])
def test_save_data_overwrite_cancelled(
    tmp_path, monkeypatch, response
):
    """Test that declining an overwrite preserves existing data."""

    filename = tmp_path / "data.npy"

    original_data = np.array([1, 2])
    np.save(filename, original_data)

    monkeypatch.setattr(
        "builtins.input",
        lambda _: response,
    )

    with patch("src.utils.helpers.toast") as mock_toast:
        result = helpers.save_data(
            np.array([3, 4]),
            str(filename),
        )

    assert result is None
    np.testing.assert_array_equal(
        np.load(filename),
        original_data,
    )

    mock_toast.assert_has_calls([
        call(
            f"File already exists:\n{filename}",
            color="orange1",
            title="OVERWRITE WARNING",
        ),
        call(
            f"Keeping existing file: {filename}",
            color="yellow",
            title="Save Cancelled",
        ),
    ])


@pytest.mark.parametrize("response", ["y", "Y", "yes", " YES "])
def test_save_data_accepts_confirmation(
    tmp_path, monkeypatch, response
):
    """Test that "y" allow saving."""

    filename = tmp_path / "data.npy"
    np.save(filename, np.array([1]))

    monkeypatch.setattr(
        "builtins.input",
        lambda _: response,
    )

    new_data = np.array([9])

    helpers.save_data(new_data, str(filename))

    np.testing.assert_array_equal(
        np.load(filename),
        new_data,
    )


# ============================================================
# load_data
# ============================================================

def test_load_data_existing_file(tmp_path):
    """Test loading data from an existing NumPy file."""

    filename = tmp_path / "data.npy"
    expected = np.array([1.0, 2.0, 3.0])

    np.save(filename, expected)

    with patch("src.utils.helpers.toast") as mock_toast:
        result = helpers.load_data(str(filename))

    np.testing.assert_array_equal(result, expected)

    mock_toast.assert_called_once_with(
        f"Loading data from: {filename}",
        color="cyan",
        title="Loading",
    )


def test_load_data_missing_file(tmp_path):
    """Test that loading a missing file raises FileNotFoundError."""

    filename = tmp_path / "missing.npy"

    with patch("src.utils.helpers.toast") as mock_toast:
        with pytest.raises(
            FileNotFoundError,
            match="Cannot load",
        ):
            helpers.load_data(str(filename))

    mock_toast.assert_called_once_with(
        f"File does not exist: {filename}",
        color="red",
        title="Load Failed",
    )


# ============================================================
# create_results_directories
# ============================================================

def test_create_results_directories(tmp_path, monkeypatch):
    """Test that the expected results directories are created."""

    monkeypatch.chdir(tmp_path)

    with patch("src.utils.helpers.toast") as mock_toast:
        helpers.create_results_directories(
            "uniaxial",
            "s_wave",
        )

    expected_dir = Path("results/uniaxial/s_wave")

    assert expected_dir.is_dir()

    mock_toast.assert_called_once_with(
        f"Results directories created:{expected_dir}",
        color="green",
        title="Directories Created",
    )


def test_create_results_directories_existing(
    tmp_path, monkeypatch
):
    """Test that creating existing directories is safe."""

    monkeypatch.chdir(tmp_path)

    expected_dir = Path("results/uniaxial/s_wave")
    expected_dir.mkdir(parents=True)

    helpers.create_results_directories(
        "uniaxial",
        "s_wave",
    )

    assert expected_dir.is_dir()


# ============================================================
# generate_data_path
# ============================================================

@pytest.mark.parametrize(
    "extension, expected_extension",
    [
        ("npy", "npy"),
        ("csv", "csv"),
        ("npz", "npz"),
    ],
)
def test_generate_data_path(extension, expected_extension):
    """Test data path generation with different extensions."""

    result = helpers.generate_data_path(
        "uniaxial",
        "s_wave",
        "hopping_array",
        extension=extension,
    )

    expected = (
        f"results/uniaxial/s_wave/"
        f"uniaxial_s_wave_hopping_array.{expected_extension}"
    )

    assert result == expected


def test_generate_data_path_default_extension():
    """Test that the default data extension is npy."""

    result = helpers.generate_data_path(
        "shear",
        "d_wave",
        "energy_band",
    )

    assert result == (
        "results/shear/d_wave/shear_d_wave_energy_band.npy"
    )


# ============================================================
# generate_plot_path
# ============================================================

def test_generate_plot_path_default_extension():
    """Test the default plot extension."""

    result = helpers.generate_plot_path(
        "uniaxial",
        "s_wave",
        "delta_vs_temperature",
    )

    assert result == (
        "results/uniaxial/s_wave/"
        "uniaxial_s_wave_delta_vs_temperature.mp4"
    )


def test_generate_plot_path_custom_extension():
    """Test a custom plot extension."""

    result = helpers.generate_plot_path(
        "shear",
        "d_wave",
        "band_structure",
        extension="gif",
    )

    assert result == (
        "results/shear/d_wave/shear_d_wave_band_structure.gif"
    )


def test_generate_plot_path_tc_always_png():
    """Test that Tc plots use PNG regardless of extension."""

    result = helpers.generate_plot_path(
        "uniaxial",
        "d_plus_id",
        "tc",
        extension="mp4",
    )

    assert result == (
        "results/uniaxial/d_plus_id/"
        "uniaxial_d_plus_id_tc.png"
    )


# ============================================================
# save_parameters_setup
# ============================================================

@pytest.fixture
def params():
    """Provide model parameters for testing parameter saving."""

    return SimpleNamespace(
        strain="uniaxial",
        gap_pairing="s_wave",
        a1x=1.0,
        a1y=0.5,
        nk=100,
        BZ_edge=np.pi,
        tx=1.0,
        tprime=0.2,
        e0=0.1,
        hopping_step=10,
        t1=0.5,
        t2=0.3,
        temperature_max=10.0,
        temperature_step=100,
        convergence_num=1000,
        conv_tolerance=1e-8,
        delta_guess=0.001,
        mixing=0.5,
        tolerance_ratio=0.01,
        Kb=8.617e-5,
    )


def test_save_parameters_setup(tmp_path, params):
    """Test that model parameters and runtimes are written."""

    filepath = tmp_path / "parameters.txt"

    with patch("src.utils.helpers.toast") as mock_toast:
        helpers.save_parameters_setup(
            filepath=str(filepath),
            params=params,
            ek_run_time=1.234,
            dos_run_time=2.345,
            delta_run_time=3.456,
            tc_run_time=4.567,
        )

    assert filepath.exists()

    content = filepath.read_text()

    assert "strain: uniaxial" in content
    assert "gap_pairing: s_wave" in content
    assert "nk: 100" in content
    assert "ek_run_time: 1.23 seconds" in content
    assert "dos_run_time: 2.35 seconds" in content
    assert "delta_run_time: 3.46 seconds" in content
    assert "tc_run_time: 4.57 seconds" in content

    mock_toast.assert_called_once_with(
        f"Parameters saved to: {filepath}",
        color="green",
        title="Parameters Saved",
    )


# ============================================================
# runtime_summary
# ============================================================

def test_runtime_summary():
    """Test runtime messages for calculated and skipped steps."""

    runtimes = [
        ("Energy band", 1.234),
        ("Density of states", 0.0),
        ("Delta", 2.5),
        ("Critical temperature", 0.0),
    ]

    with patch("src.utils.helpers.toast") as mock_toast:
        helpers.runtime_summary(runtimes)

    assert mock_toast.call_args_list == [
        call(
            "Energy band: 1.23 seconds",
            color="green",
            title="Runtime Summary",
        ),
        call(
            "Density of states not calculated.",
            color="cyan",
            title="Warning",
        ),
        call(
            "Delta: 2.50 seconds",
            color="green",
            title="Runtime Summary",
        ),
        call(
            "Critical temperature not calculated.",
            color="cyan",
            title="Warning",
        ),
    ]


def test_runtime_summary_empty():
    """Test that an empty runtime list produces no messages."""

    with patch("src.utils.helpers.toast") as mock_toast:
        helpers.runtime_summary([])

    mock_toast.assert_not_called()
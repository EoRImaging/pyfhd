from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
import h5py
from logging import Logger
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.healpix.healpix_utils import vis_model_freq_split


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "healpix", "vis_model_freq_split")


@pytest.fixture
def beam_file():
    return Path(env.get("PYFHD_TEST_PATH"), "beams", "decomp_beam_pointing0.h5")


@pytest.fixture(
    scope="function",
    params=["1088285600", "1088716296", "point_zenith", "point_offzenith"],
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3", "run4"])
def run(request):
    return request.param


skip_tests = [
    ["1088285600", "run4"],
    ["point_zenith", "run3"],
    ["point_offzenith", "run3"],
    ["1088716296", "run3"],
]


@pytest.fixture
def before_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    pyfhd_config = {
        "interpolate_kernel": sav_dict["extra"]["interpolate_kernel"],
        "mask_mirror_indices": False,
        "beam_per_baseline": False,
        "grid_uniform": False,
        "grid_spectral": False,
        "grid_weights": True,
        "grid_variance": True,
        "n_avg": sav_dict["n_avg"],
        "rephase_weights": sav_dict["rephase_weights"],
        "image_filter": "filter_uv_uniform",
        "beam_clip_floor": True,
    }

    sav_dict["pyfhd_config"] = pyfhd_config

    del sav_dict["extra"]
    del sav_dict["n_avg"]
    del sav_dict["rephase_weights"]

    # Swap the baselines and frequencies around
    sav_dict["vis_weights"] = np.swapaxes(sav_dict["vis_weights"], -2, -1)
    sav_dict["vis_model_arr"] = np.swapaxes(sav_dict["vis_model_arr"], -2, -1)
    sav_dict["vis_data_arr"] = np.swapaxes(sav_dict["vis_data_arr"], -2, -1)

    save(before_file, sav_dict, "b5_before")

    return before_file


@pytest.fixture()
def after_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)
    # Swap the frequencies and polarization around, take the first polarization
    sav_dict["dirty_arr"] = np.swapaxes(sav_dict["dirty_arr"], 0, 1)[0]
    sav_dict["model_arr"] = np.swapaxes(sav_dict["model_arr"], 0, 1)[0]
    sav_dict["weights_arr"] = np.swapaxes(sav_dict["weights_arr"], 0, 1)[0]
    sav_dict["variance_arr"] = np.swapaxes(sav_dict["variance_arr"], 0, 1)[0]

    save(after_file, sav_dict, "b5_after")

    return after_file


def test_vis_model_freq_split(before_file, after_file, beam_file):
    pytest.skip("Skipped due to the system needing more memory than it had available.")
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    expected_model_split = load(after_file, lazy_load=True)

    psf = load(beam_file, lazy_load=True)

    h5_before["obs"]["dimension"] = int(h5_before["obs"]["dimension"])
    h5_before["obs"]["elements"] = int(h5_before["obs"]["elements"])
    h5_before["obs"]["n_baselines"] = h5_before["obs"]["nbaselines"]

    model_split = vis_model_freq_split(
        h5_before["obs"],
        psf,
        h5_before["params"],
        h5_before["vis_weights"],
        h5_before["vis_model_arr"],
        h5_before["vis_data_arr"],
        0,
        h5_before["pyfhd_config"],
        Logger("test"),
        fft=h5_before["fft"],
        save_uvf=False,
        uvf_name=h5_before["uvf_name"],
        bi_use=h5_before["bi_use"],
    )

    assert expected_model_split["obs_out"]["n_vis"] == model_split["obs"]["n_vis"]
    # Only checking the first polarization due to the size of the arrays taking up too
    # much memory, now doing the split on a per polarization basis due to memory constraints
    npt.assert_allclose(
        model_split["residual_arr"],
        expected_model_split["dirty_arr"],
        atol=1e-8,
    )
    npt.assert_allclose(
        model_split["weights_arr"],
        expected_model_split["weights_arr"],
        atol=1e-8,
    )
    npt.assert_allclose(
        model_split["variance_arr"],
        expected_model_split["variance_arr"],
        atol=1e-8,
    )
    npt.assert_allclose(
        model_split["model_arr"],
        expected_model_split["model_arr"],
        atol=1e-8,
    )

from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.calibration.calibration_utils import vis_cal_polyfit
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
import numpy.testing as npt
from pyfhd.io.pyfhd_io import save, load
import numpy as np
import importlib_resources
from logging import Logger


@pytest.fixture
def data_dir():
    return importlib_resources.files("pyfhd.resources.test_data").joinpath(
        "calibration", "vis_cal_polyfit"
    )


@pytest.fixture(
    scope="function", params=["point_zenith", "point_offzenith", "1088716296"]
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run1", "run2", "run3"])
def run(request):
    return request.param


skip_tests = [["1088716296", "run3"]]


@pytest.fixture()
def before_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    obs = recarray_to_dict(sav_dict["obs"])
    cal = recarray_to_dict(sav_dict["cal"])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    gain = sav_file_vis_arr_swap_axes(cal["gain"])
    cal["gain"] = gain

    fhd_keys = [
        "cal_reflection_mode_theory",
        "cal_reflection_mode_file",
        "cal_reflection_mode_delay",
        "cal_reflection_hyperresolve",
        "amp_degree",
        "phase_degree",
    ]
    config_keys = [
        "cal_reflection_mode_theory",
        "cal_reflection_mode_file",
        "cal_reflection_mode_delay",
        "cal_reflection_hyperresolve",
        "cal_amp_degree_fit",
        "cal_phase_degree_fit",
    ]

    # make a slimmed down version of pyfhd_config
    pyfhd_config = {}

    # When keys are unset in IDL, they just don't save to a .sav
    # file. So try accessing with an exception and set to None
    # if they don't exists
    for fhd_key, config_key in zip(fhd_keys, config_keys):
        try:
            pyfhd_config[config_key] = sav_dict[fhd_key]
        except KeyError:
            pyfhd_config[config_key] = None

    if "digital_gain_jump_polyfit" in sav_dict:
        pyfhd_config["digital_gain_jump_polyfit"] = sav_dict[
            "digital_gain_jump_polyfit"
        ]
    else:
        pyfhd_config["digital_gain_jump_polyfit"] = True
    if "auto_ratio" in sav_dict:
        auto_ratio = sav_file_vis_arr_swap_axes(sav_dict["auto_ratio"])
    else:
        auto_ratio = None

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = obs
    h5_save_dict["cal"] = cal
    h5_save_dict["pyfhd_config"] = pyfhd_config
    h5_save_dict["auto_ratio"] = auto_ratio
    del h5_save_dict["cal"]["amp_params"]
    del h5_save_dict["cal"]["phase_params"]
    del h5_save_dict["cal"]["mode_params"]

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture()
def after_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.sav")

    sav_dict = convert_sav_to_dict(str(before_file), "faked")
    obs = recarray_to_dict(sav_dict["obs"])

    fhd_keys = ["amp_degree", "phase_degree"]
    config_keys = ["cal_amp_degree_fit", "cal_phase_degree_fit"]

    # make a slimmed down version of pyfhd_config
    pyfhd_config = {}

    # When keys are unset in IDL, they just don't save to a .sav
    # file. So try accessing with an exception and set to None
    # if they don't exists
    for fhd_key, config_key in zip(fhd_keys, config_keys):
        try:
            pyfhd_config[config_key] = sav_dict[fhd_key]
        except KeyError:
            pyfhd_config[config_key] = None

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    cal_return = recarray_to_dict(sav_dict["cal_return"])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    gain = sav_file_vis_arr_swap_axes(cal_return["gain"])
    cal_return["gain"] = gain

    # cal_return keeps amp_params as a pointer array of shape (128, 2)
    # However because digital_gain_jump_polyfit was used each pointer contains a (2,2)
    # This means the shape should be (2, 128, 2, 2) or
    # (n_pol, n_tile, cal_amp_degree_fit, cal_amp_degree_fit)
    # cal_return also keeps the phase_params as (128, 2) object array, each one containing two 1 element float arrays
    # This should be (n_pol, n_tile, cal_phase_degree_fit + 1) or (2, 128, 2) so
    # We'll grab each one by tile and polarization, and stack and flatten.
    expected_amp_params = np.empty(
        (
            cal_return["n_pol"],
            obs["n_tile"],
            pyfhd_config["cal_amp_degree_fit"],
            pyfhd_config["cal_amp_degree_fit"],
        )
    )
    expected_phase_params = np.empty(
        (cal_return["n_pol"], obs["n_tile"], pyfhd_config["cal_phase_degree_fit"] + 1)
    )
    expected_mode_params = np.empty((cal_return["n_pol"], obs["n_tile"], 3))
    # Recarray didn't know what to do with it, so turn it back into object array
    cal_return["mode_params"] = np.array(cal_return["mode_params"])
    # Convert the amp, phase and mode params to not object arrays, or put them in the right size
    for pol_i in range(cal_return["n_pol"]):
        for tile_i in range(obs["n_tile"]):
            expected_amp_params[pol_i, tile_i] = np.transpose(
                cal_return["amp_params"][tile_i, pol_i]
            )
            expected_phase_params[pol_i, tile_i] = np.vstack(
                cal_return["phase_params"][tile_i, pol_i]
            ).flatten()
            if cal_return["phase_params"][tile_i, pol_i][0] is None:
                expected_mode_params[pol_i, tile_i] = np.full(
                    3, np.nan, dtype=np.float64
                )
            else:
                expected_mode_params[pol_i, tile_i] = cal_return["mode_params"][
                    tile_i, pol_i
                ]
    cal_return["amp_params"] = expected_amp_params
    cal_return["phase_params"] = expected_phase_params
    cal_return["mode_params"] = expected_mode_params

    save(after_file, cal_return, "after_file")

    return after_file


@pytest.mark.github_actions
def test_vis_cal_polyfit(before_file, after_file):
    """Runs the test on `vis_cal_polyfit` - reads in the data in before_file and after_file,
    and then calls `vis_cal_polyfit`, checking the outputs match expectations"""
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    expected_cal_return = load(after_file)

    obs = h5_before["obs"]
    cal = h5_before["cal"]
    auto_ratio = h5_before["auto_ratio"]

    pyfhd_config = h5_before["pyfhd_config"]
    pyfhd_config["instrument"] = "mwa"
    logger = Logger(1)

    cal_polyfit, _ = vis_cal_polyfit(obs, cal, auto_ratio, pyfhd_config, logger)
    npt.assert_allclose(
        cal_polyfit["amp_params"], expected_cal_return["amp_params"], atol=2e-7
    )
    # Only the real data has atol error of 2e-6, simulated has 1e-8 atol error
    npt.assert_allclose(
        cal_polyfit["phase_params"], expected_cal_return["phase_params"], atol=2e-6
    )
    # atol due to differences in precision differences in multiple places with multiplication and polyfits
    # of single precision
    npt.assert_allclose(cal_polyfit["gain"], expected_cal_return["gain"], atol=2e-5)
    # Test the mode params ignoring the nans from FHD
    # non_nans = ~np.isnan(expected_cal_return['mode_params'])
    # Slight differences in mode indexes make this not possible to test properly
    # npt.assert_allclose(cal_polyfit['mode_params'][non_nans], expected_cal_return['mode_params'][non_nans], atol=1e-8)

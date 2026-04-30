from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.calibration.calibration_utils import vis_cal_bandpass
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
import numpy as np
from pyfhd.io.pyfhd_io import save, load
from logging import Logger
import numpy.testing as npt
import importlib_resources


@pytest.fixture
def data_dir():
    return importlib_resources.files("pyfhd.resources.test_data").joinpath(
        "calibration", "vis_cal_bandpass"
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

    obs = recarray_to_dict(sav_dict["obs"])
    cal = recarray_to_dict(sav_dict["cal"])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    gain = sav_file_vis_arr_swap_axes(cal["gain"])
    cal["gain"] = gain

    # make a slimmed down version of pyfhd_config
    pyfhd_config = {}

    # If keywords cal_bp_transfer, cable_bandpass_fit are not set, IDL won't save it,
    # so catch non-existent key and set to zero if this is the case
    # Also, pyfhd uses None instead of 0 for False, so swap to none if
    # needed
    try:
        pyfhd_config["cal_bp_transfer"] = sav_dict["cal_bp_transfer"]
        if pyfhd_config["cal_bp_transfer"] == 0:
            pyfhd_config["cal_bp_transfer"] = None
    except KeyError:
        pyfhd_config["cal_bp_transfer"] = None

    try:
        pyfhd_config["cable_bandpass_fit"] = sav_dict["cable_bandpass_fit"]
    except KeyError:
        pyfhd_config["cable_bandpass_fit"] = False

    try:
        pyfhd_config["auto_ratio_calibration"] = sav_dict["auto_ratio_calibration"]
    except KeyError:
        pyfhd_config["auto_ratio_calibration"] = False

    # need the instrument as that's needed for a file path
    pyfhd_config["instrument"] = "mwa"

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = obs
    h5_save_dict["cal"] = cal
    h5_save_dict["pyfhd_config"] = pyfhd_config

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

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    cal_bandpass = recarray_to_dict(sav_dict["cal_bandpass"])
    cal_remainder = recarray_to_dict(sav_dict["cal_remainder"])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    gain = sav_file_vis_arr_swap_axes(cal_bandpass["gain"])
    cal_bandpass["gain"] = gain

    gain = sav_file_vis_arr_swap_axes(cal_remainder["gain"])
    cal_remainder["gain"] = gain

    # super dictionary to save everything in
    h5_save_dict = {}

    h5_save_dict["cal_bandpass"] = cal_bandpass
    h5_save_dict["cal_remainder"] = cal_remainder

    save(after_file, h5_save_dict, "after_file")

    return after_file


@pytest.mark.github_actions
def test_vis_cal_bandpass(before_file, after_file):
    """Runs the test on `vis_cal_bandpass` - reads in the data in `data_loc`,
    and then calls `vis_cal_bandpass`, checking the outputs match expectations"""
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    h5_after = load(after_file)

    obs = h5_before["obs"]
    cal = h5_before["cal"]
    pyfhd_config = h5_before["pyfhd_config"]
    # pyfhd_config['cal_bp_transfer'] = None

    expected_cal_bandpass = h5_after["cal_bandpass"]
    expected_cal_remainder = h5_after["cal_remainder"]

    logger = Logger(1)

    result_cal_bandpass, result_cal_remainder = vis_cal_bandpass(
        obs, cal, pyfhd_config, logger
    )

    # The FHD function does some dividing by zeros, so we end up with NaNs
    # in both the expected and result data. To check we are replicating
    # the NaNs correctly, check both results for NaNs and assert they are
    # in the same place

    expec_nan_inds = np.where(np.isnan(expected_cal_remainder["gain"]) == True)
    result_nan_inds = np.where(np.isnan(expected_cal_remainder["gain"]) == True)

    npt.assert_array_equal(expec_nan_inds, result_nan_inds)

    # find where things are not NaN and check they are close
    test_inds = np.where(np.isnan(expected_cal_remainder["gain"]) == False)

    rtol = 5e-5
    atol = 1e-8

    npt.assert_allclose(
        expected_cal_remainder["gain"][test_inds],
        result_cal_remainder["gain"][test_inds],
        rtol=rtol,
        atol=atol,
    )

    # shouoldn't be NaNs in this, so just check all the outputs
    npt.assert_allclose(
        expected_cal_bandpass["gain"], result_cal_bandpass["gain"], rtol=rtol, atol=atol
    )

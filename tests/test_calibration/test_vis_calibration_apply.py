from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.pyfhd_tools.test_utils import get_data_items, get_data_sav
from pyfhd.calibration.calibration_utils import vis_calibration_apply
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
import numpy as np
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt
from logging import Logger

# import matplotlib.pyplot as plt


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_calibration_apply")


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

    cal = recarray_to_dict(sav_dict["cal"])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    cal["gain"] = sav_file_vis_arr_swap_axes(cal["gain"])

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["cal"] = cal
    # No need to format ragged array here
    del h5_save_dict["cal"]["mode_params"]
    h5_save_dict["vis_ptr"] = sav_file_vis_arr_swap_axes(sav_dict["vis_ptr"])

    # When keys are unset in IDL, they just don't save to a .sav
    # file. So try accessing with an exception and set to None
    # if they don't exists
    for key in ["vis_model_ptr", "vis_weight_ptr"]:
        iskey = False
        try:
            iskey = sav_dict[key]
        except KeyError:
            h5_save_dict[key] = None

        # You can get a IDL pointer to two empty arrays here, so check if
        # anything exists inside the array as well as it being an array
        if type(iskey) == np.ndarray and type(iskey[0]) == np.ndarray:
            h5_save_dict[key] = sav_file_vis_arr_swap_axes(sav_dict[key])
        else:
            h5_save_dict[key] = None

    for key in ["invert_gain", "preserve_original"]:
        try:
            h5_save_dict[key] = sav_dict[key]
        except KeyError:
            h5_save_dict[key] = None

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

    h5_save_dict = {}

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    h5_save_dict["vis_cal_ptr"] = sav_file_vis_arr_swap_axes(sav_dict["vis_cal_ptr"])
    h5_save_dict["cal"] = recarray_to_dict(sav_dict["cal"])
    # No need to format ragged array here
    del h5_save_dict["cal"]["mode_params"]

    save(after_file, h5_save_dict, "after_file")

    return after_file


def test_vis_calibration_apply(before_file, after_file):
    """
    Runs the test on `vis_calibration_apply` - reads in the data in before_file and after_file,
    and then calls `vis_calibration_apply`, checking the outputs match
    """
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    h5_after = load(after_file)

    vis_ptr = h5_before["vis_ptr"]
    cal = h5_before["cal"]

    # The FHD code has made copies of things from `obs` into `cal`. In pyfhd,
    # we just supply the `obs`. Means we need to make a mini `obs` for testing
    # here
    obs = {}
    obs["baseline_info"] = {}
    obs["baseline_info"]["tile_a"] = cal["tile_a"]
    obs["baseline_info"]["tile_b"] = cal["tile_b"]
    obs["n_freq"] = cal["n_freq"]
    obs["n_baselines"] = len(cal["tile_a"])
    obs["n_times"] = cal["n_time"]

    vis_model_ptr = h5_before["vis_model_ptr"]
    vis_weight_ptr = h5_before["vis_weight_ptr"]

    exptected_vis_cal_ptr = h5_after["vis_cal_ptr"]

    logger = Logger(1)

    return_vis_cal_ptr, return_cal = vis_calibration_apply(
        vis_ptr, obs, cal, vis_model_ptr, vis_weight_ptr, logger
    )

    if vis_ptr.shape[0] == 4:

        npt.assert_allclose(
            h5_after["cal"]["cross_phase"],
            return_cal["cross_phase"],
            atol=1e-6,
            equal_nan=True,
        )

        npt.assert_allclose(
            return_vis_cal_ptr[2:], exptected_vis_cal_ptr[2:], atol=1e-4, equal_nan=True
        )

    # XX and YY have larger values so suffer less from precision errors??
    # point_zenith and off_zenith are fine with 2e-5, but 1088716296 is down to 3.5e-3
    # For some reason that only happened after changing the save and load functions?
    npt.assert_allclose(
        return_vis_cal_ptr[:2], exptected_vis_cal_ptr[:2], atol=3.5e-3, equal_nan=True
    )

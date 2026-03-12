from PyFHD.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from PyFHD.calibration.calibrate import calibrate_qu_mixing
from PyFHD.io.pyfhd_io import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from PyFHD.io.pyfhd_io import save, load
import numpy.testing as npt


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_calibrate_qu_mixing")


@pytest.fixture(scope="function", params=["point_zenith", "point_offzenith"])
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3"])
def run(request):
    return request.param


# skip_tests = [['point_zenith', 'run3']]
skip_tests = []


@pytest.fixture
def before_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])
    h5_save_dict["vis_ptr"] = sav_file_vis_arr_swap_axes(sav_dict["vis_ptr"])
    h5_save_dict["vis_model_ptr"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_model_ptr"]
    )
    h5_save_dict["vis_weight_ptr"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_weight_ptr"]
    )

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture
def after_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["calc_phase"] = sav_dict["calc_phase"]

    save(after_file, h5_save_dict, "after_file")

    return after_file


def test_qu_mixing(before_file, after_file):
    """Runs the test on `calibrate_qu_mixing` - reads in the data in `data_loc`,
    and then calls `calibrate_qu_mixing`, checking the outputs match expectations"""
    if before_file == None or after_file == None:
        pytest.skip(f""""This test has been skipped because the test was listed 
                    in the skipped tests due to FHD not outputting them: {skip_tests}.
                    In this case it as due to the LA_LEAST_SQUARES differences
                    compared to np.linalg.lstsq which doesn't use double precision
                    by default which for some reason makes a difference for values
                    close to 0 in single precision.""")

    h5_before = load(before_file)
    expected_calc_phase = load(after_file)

    obs = h5_before["obs"]
    obs["n_baselines"] = obs["nbaselines"]
    vis_ptr = h5_before["vis_ptr"]
    vis_model_ptr = h5_before["vis_model_ptr"]
    vis_weight_ptr = h5_before["vis_weight_ptr"]

    result_cal_phase = calibrate_qu_mixing(vis_ptr, vis_model_ptr, vis_weight_ptr, obs)
    # Higher error due to LA_LEAST_SQUARES not using double precision
    # by default which for some reason makes a difference for values
    # close to 0 in single precision.
    atol = 3e-4

    npt.assert_allclose(expected_calc_phase, result_cal_phase, atol=atol)

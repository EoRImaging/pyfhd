from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from pyfhd.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from glob import glob
from logging import Logger
from pyfhd.pyfhd_tools.test_utils import get_data_items, sav_file_vis_arr_swap_axes
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from numpy.testing import assert_allclose
from pyfhd.io.pyfhd_io import save, load
import importlib_resources


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_calibrate_subroutine")


@pytest.fixture(
    scope="function",
    params=["point_zenith", "point_offzenith", "1088716296", "1088285600"],
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run1", "run2", "run3"])
def run(request):
    return request.param


skip_tests = [["1088716296", "run3"], ["1088285600", "run3"]]


@pytest.fixture()
def before_file(tag, run, data_dir, request: pytest.FixtureRequest):
    if [tag, run] in skip_tests:
        return None
    if tag == "point_zenith" and run == "run1":
        request.node.add_marker(pytest.mark.github_actions)
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "calibration", "vis_calibrate_subroutine"
        )
        # Use the one of the files that has been created as before file
        before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}_cal.h5")
    else:
        before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    h5_save_dict = {}
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])
    h5_save_dict["cal"] = recarray_to_dict(sav_dict["cal"])
    h5_save_dict["cal"]["gain"] = sav_file_vis_arr_swap_axes(
        h5_save_dict["cal"]["gain"]
    )
    # Since we don't want to copy data we are leaving uu and vv in params in pyfhd
    # Thus we need to copy across for the test and save params as a separate dict
    params = {}
    params["uu"] = h5_save_dict["cal"]["uu"]
    params["vv"] = h5_save_dict["cal"]["vv"]
    h5_save_dict["params"] = params
    h5_save_dict["vis_ptr"] = sav_file_vis_arr_swap_axes(sav_dict["vis_ptr"])
    h5_save_dict["vis_model_ptr"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_model_ptr"]
    )
    h5_save_dict["vis_weight_ptr"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_weight_ptr"]
    )
    # Simulate pyfhd_config
    pyfhd_config = {}
    pyfhd_config["min_cal_baseline"] = h5_save_dict["cal"]["min_cal_baseline"]
    pyfhd_config["max_cal_baseline"] = h5_save_dict["cal"]["max_cal_baseline"]
    pyfhd_config["cal_time_average"] = h5_save_dict["cal"]["time_avg"]
    pyfhd_config["cal_adaptive_calibration_gain"] = h5_save_dict["cal"]["adaptive_gain"]
    pyfhd_config["cal_convergence_threshold"] = h5_save_dict["cal"]["conv_thresh"]
    pyfhd_config["cal_base_gain"] = h5_save_dict["cal"]["base_gain"]
    pyfhd_config["cal_phase_fit_iter"] = h5_save_dict["cal"]["phase_iter"]
    pyfhd_config["max_cal_iter"] = h5_save_dict["cal"]["max_iter"]
    h5_save_dict["pyfhd_config"] = pyfhd_config

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture()
def after_file(tag, run, data_dir, request: pytest.FixtureRequest):
    if [tag, run] in skip_tests:
        return None
    if tag == "point_zenith" and run == "run1":
        request.node.add_marker(pytest.mark.github_actions)
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "calibration", "vis_calibrate_subroutine"
        )
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    cal_return = recarray_to_dict(sav_dict["cal_return"])
    cal_return["gain"] = sav_file_vis_arr_swap_axes(cal_return["gain"])

    save(after_file, cal_return, "after_file")

    return after_file


def test_points(before_file, after_file):
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )
    if "point_zenith_run1" in str(before_file):
        cal = load(before_file)
        obs = load(str(before_file).replace("_cal.h5", "_obs.h5"))
        params = load(str(before_file).replace("_cal.h5", "_params.h5"))
        pyfhd_config = load(str(before_file).replace("_cal.h5", "_pyfhd_config.h5"))
        vis_ptr_xx = load(str(before_file).replace("_cal.h5", "_vis_ptr_XX.h5"))
        vis_ptr_yy = load(str(before_file).replace("_cal.h5", "_vis_ptr_YY.h5"))
        vis_model_ptr_xx = load(
            str(before_file).replace("_cal.h5", "_vis_model_ptr_XX.h5")
        )
        vis_model_ptr_yy = load(
            str(before_file).replace("_cal.h5", "_vis_model_ptr_YY.h5")
        )
        vis_weight_ptr_xx = load(
            str(before_file).replace("_cal.h5", "_vis_weight_ptr_XX.h5")
        )
        vis_weight_ptr_yy = load(
            str(before_file).replace("_cal.h5", "_vis_weight_ptr_YY.h5")
        )
        h5_before = {
            "vis_ptr": np.array([vis_ptr_xx, vis_ptr_yy]),
            "vis_model_ptr": np.array([vis_model_ptr_xx, vis_model_ptr_yy]),
            "vis_weight_ptr": np.array([vis_weight_ptr_xx, vis_weight_ptr_yy]),
            "obs": obs,
            "cal": cal,
            "params": params,
            "pyfhd_config": pyfhd_config,
        }
    else:
        h5_before = load(before_file)
    expected_cal = load(after_file)

    vis_ptr = h5_before["vis_ptr"]
    vis_model_ptr = h5_before["vis_model_ptr"]
    vis_weight_ptr = h5_before["vis_weight_ptr"]
    obs = h5_before["obs"]
    obs["n_baselines"] = obs["nbaselines"]
    cal = h5_before["cal"]
    params = h5_before["params"]
    pyfhd_config = h5_before["pyfhd_config"]

    logger = Logger(1)

    cal_return = vis_calibrate_subroutine(
        vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, params, pyfhd_config, logger
    )

    assert expected_cal["n_vis_cal"] == cal_return["n_vis_cal"]
    # Code was added that flagged the gain based on tiles, making them NaN, FHD doesn't do this
    # So to compare we need to remove the NaN values
    assert_allclose(
        cal_return["gain"][:, :, ~cal_return["tile_flag"]],
        expected_cal["gain"][:, :, ~cal_return["tile_flag"]],
        atol=4e-05,
    )


@pytest.fixture(scope="function", params=[1, 2, 3])
def subroutine_test(request):
    return request.param


@pytest.fixture
def subroutine_before(data_dir, subroutine_test):
    subroutine_before = Path(
        data_dir, f"test_{subroutine_test}_before_{data_dir.name}.h5"
    )

    if subroutine_before.exists():
        return subroutine_before

    vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal = get_data_items(
        data_dir,
        f"input_vis_ptr_{subroutine_test}.npy",
        f"input_vis_model_ptr_{subroutine_test}.npy",
        f"input_vis_weight_ptr_{subroutine_test}.npy",
        f"input_obs_{subroutine_test}.npy",
        f"input_cal_{subroutine_test}.npy",
    )

    h5_save_dict = {}
    h5_save_dict["vis_ptr"] = sav_file_vis_arr_swap_axes(vis_ptr)
    h5_save_dict["vis_model_ptr"] = sav_file_vis_arr_swap_axes(vis_model_ptr)
    h5_save_dict["vis_weight_ptr"] = sav_file_vis_arr_swap_axes(vis_weight_ptr)
    h5_save_dict["obs"] = recarray_to_dict(obs)
    h5_save_dict["cal"] = recarray_to_dict(cal)
    h5_save_dict["cal"]["gain"] = sav_file_vis_arr_swap_axes(
        h5_save_dict["cal"]["gain"]
    )
    h5_save_dict["calibration_weights"] = False
    if subroutine_test == 2:
        h5_save_dict["calibration_weights"] = True
    # Simulate params
    params = {}
    params["uu"] = h5_save_dict["cal"]["uu"]
    params["vv"] = h5_save_dict["cal"]["vv"]
    h5_save_dict["params"] = params
    # Simulate pyfhd_config
    pyfhd_config = {}
    pyfhd_config["min_cal_baseline"] = h5_save_dict["cal"]["min_cal_baseline"]
    pyfhd_config["max_cal_baseline"] = h5_save_dict["cal"]["max_cal_baseline"]
    pyfhd_config["cal_time_average"] = h5_save_dict["cal"]["time_avg"]
    pyfhd_config["cal_adaptive_calibration_gain"] = h5_save_dict["cal"]["adaptive_gain"]
    pyfhd_config["cal_convergence_threshold"] = h5_save_dict["cal"]["conv_thresh"]
    pyfhd_config["cal_base_gain"] = h5_save_dict["cal"]["base_gain"]
    pyfhd_config["cal_phase_fit_iter"] = h5_save_dict["cal"]["phase_iter"]
    pyfhd_config["max_cal_iter"] = h5_save_dict["cal"]["max_iter"]
    h5_save_dict["pyfhd_config"] = pyfhd_config

    save(subroutine_before, h5_save_dict, "before_file")

    return subroutine_before


@pytest.fixture
def subroutine_after(data_dir, subroutine_test):
    subroutine_after = Path(
        data_dir, f"test_{subroutine_test}_after_{data_dir.name}.h5"
    )

    if subroutine_after.exists():
        return subroutine_after

    cal_return = recarray_to_dict(
        get_data_items(data_dir, f"output_cal_return_{subroutine_test}.npy")
    )
    cal_return["gain"] = sav_file_vis_arr_swap_axes(cal_return["gain"])

    save(subroutine_after, cal_return, "after_file")

    return subroutine_after


def test_vis_calibration_x(subroutine_before, subroutine_after):

    h5_before = load(subroutine_before)
    expected_cal_return = load(subroutine_after)

    vis_ptr = h5_before["vis_ptr"]
    vis_model_ptr = h5_before["vis_model_ptr"]
    vis_weight_ptr = h5_before["vis_weight_ptr"]
    obs = h5_before["obs"]
    obs["n_baselines"] = obs["nbaselines"]
    cal = h5_before["cal"]
    calibration_weights = h5_before["calibration_weights"]
    params = h5_before["params"]
    pyfhd_config = h5_before["pyfhd_config"]

    logger = Logger(1)

    cal_return = vis_calibrate_subroutine(
        vis_ptr,
        vis_model_ptr,
        vis_weight_ptr,
        obs,
        cal,
        params,
        pyfhd_config,
        logger,
        calibration_weights=calibration_weights,
    )

    assert expected_cal_return["n_vis_cal"] == cal_return["n_vis_cal"]

    assert_allclose(
        cal_return["gain"][:, :, ~cal_return["tile_flag"]],
        expected_cal_return["gain"][:, :, ~cal_return["tile_flag"]],
        atol=1.02e-03,
    )

from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from logging import Logger
from pathlib import Path
from os import environ as env
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from pyfhd.flagging.flagging import vis_flag_basic
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt
import numpy as np


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "flagging", "vis_flag_basic")


@pytest.fixture(
    scope="function", params=["1088716296", "point_zenith", "point_offzenith"]
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run1", "run2", "run3"])
def run(request):
    return request.param


skip_tests: list = []


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

    h5_save_dict = {}
    h5_save_dict["vis_weights"] = sav_file_vis_arr_swap_axes(sav_dict["vis_weights"])
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])

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
    h5_save_dict["vis_weights"] = sav_file_vis_arr_swap_axes(sav_dict["vis_weights"])
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])

    save(after_file, h5_save_dict, "after_file")

    return after_file


def test_many_points(before_file, after_file):
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    logger = Logger(1)

    h5_before = load(before_file)
    h5_after = load(after_file)

    vis_weight_arr = h5_before["vis_weights"]
    obs = h5_before["obs"]

    pyfhd_config = {
        "flag_freq_start": None,
        "flag_freq_end": None,
        "flag_tiles": [],
        "flag_frequencies": True,
    }

    # Ran out of time to implement the flag_frequencies being false test case
    faked_vis_arr = np.zeros(vis_weight_arr.shape, dtype=vis_weight_arr.dtype)

    vis_weights_result, obs_result = vis_flag_basic(
        vis_weight_arr, faked_vis_arr, obs, pyfhd_config, logger
    )

    npt.assert_allclose(vis_weights_result, h5_after["vis_weights"])
    npt.assert_array_equal(
        obs_result["baseline_info"]["tile_use"],
        h5_after["obs"]["baseline_info"]["tile_use"],
    )
    npt.assert_array_equal(
        obs_result["baseline_info"]["freq_use"],
        h5_after["obs"]["baseline_info"]["freq_use"],
    )
    assert obs_result["n_time_flag"] == h5_after["obs"]["n_time_flag"]
    assert obs_result["n_tile_flag"] == h5_after["obs"]["n_tile_flag"]
    assert obs_result["n_freq_flag"] == h5_after["obs"]["n_freq_flag"]

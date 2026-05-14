from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.pyfhd_tools.pyfhd_utils import vis_noise_calc
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
import numpy as np
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_noise_calc")


@pytest.fixture(scope="function", params=["point_zenith", "point_offzenith"])
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3"])
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

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])
    h5_save_dict["vis_arr"] = sav_file_vis_arr_swap_axes(sav_dict["vis_arr"])
    h5_save_dict["vis_weights"] = sav_file_vis_arr_swap_axes(sav_dict["vis_weights"])
    # print('vis-weights shape', h5_save_dict['vis_weights'].shape)

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

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["noise_arr"] = sav_dict["noise_arr"]

    save(after_file, sav_dict["noise_arr"].transpose(), "noise_arr")

    return after_file


def test_points_zenith_and_offzenith(before_file, after_file):
    """Runs the test on `vis_noise_calc` - reads in the data in `data_loc`,
    and then calls `vis_noise_calc`, checking the outputs match expectations"""

    h5_before = load(before_file)
    expected_noise_arr = load(after_file)

    obs = h5_before["obs"]
    vis_arr = h5_before["vis_arr"]
    vis_weights = h5_before["vis_weights"]

    result_noise_arr = vis_noise_calc(obs, vis_arr, vis_weights)

    # IDL Stddev returns NaN for some values on single precision, but not on double precision, compare only non NaN.
    if np.any(np.isnan(expected_noise_arr)):
        not_nan_idxs = np.where(~np.isnan(expected_noise_arr))
        npt.assert_allclose(
            expected_noise_arr[not_nan_idxs], result_noise_arr[not_nan_idxs], atol=1e-10
        )
    else:
        npt.assert_allclose(expected_noise_arr, result_noise_arr, atol=3e-5)

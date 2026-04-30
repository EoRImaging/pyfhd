from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.pyfhd_tools.pyfhd_utils import split_vis_weights
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "split_vis_weights")


@pytest.fixture(scope="function", params=["point_zenith", "point_offzenith"])
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3"])
def run(request):
    return request.param


@pytest.fixture
def before_file(data_dir, tag, run):
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])
    h5_save_dict["vis_weights"] = sav_file_vis_arr_swap_axes(sav_dict["vis_weights"])

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture
def after_file(data_dir, tag, run):
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file
    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["vis_weights_use"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_weights_use"]
    )
    h5_save_dict["bi_use"] = sav_dict["bi_use"]
    # Format the dict appropriately
    h5_save_dict = recarray_to_dict(h5_save_dict)

    save(after_file, h5_save_dict, "after_file")

    return after_file


def test_split_vis_weights(before_file, after_file):
    h5_before = load(before_file)
    h5_after = load(after_file)

    obs = h5_before["obs"]
    vis_weights = h5_before["vis_weights"]

    expected_vis_weights = h5_after["vis_weights_use"]
    expected_bi_use = h5_after["bi_use"]

    result_vis_weights, result_bi_use = split_vis_weights(obs, vis_weights)

    atol = 1e-8

    # Check the returned weights have been spli correctly
    npt.assert_allclose(expected_vis_weights, result_vis_weights, atol=atol)

    # Annoying shape mis-match, so test each polarisation result individually
    for pol in range(2):
        npt.assert_allclose(expected_bi_use[pol], result_bi_use[pol], atol=atol)

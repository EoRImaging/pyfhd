import pytest
from os import environ as env
from pathlib import Path
from pyfhd.io.pyfhd_io import save, load
import numpy as np

from pyfhd.calibration.calibration_utils import vis_extract_autocorr
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.io.pyfhd_io import recarray_to_dict


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_extract_autocorr")


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

    # convert obs from a recarray into a dictionary
    obs = recarray_to_dict(sav_dict["obs"])

    # fix the shape of vis_arr
    # inside vis_extract_autocorr
    # Currently I'm reading into a shape of (n_pol, n_freq, n_baselines)
    vis_arr = np.empty(
        (
            obs["n_pol"],
            sav_dict["vis_arr"][0].shape[1],
            sav_dict["vis_arr"][0].shape[0],
        ),
        dtype=complex,
    )
    for pol in range(obs["n_pol"]):
        vis_arr[pol, :, :] = sav_dict["vis_arr"][pol].transpose()

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = obs
    h5_save_dict["vis_arr"] = vis_arr
    pyfhd_config = {}
    if "time_average" in sav_dict:
        pyfhd_config["cal_time_average"] = sav_dict["time_average"]
    else:
        # Set to default
        pyfhd_config["cal_time_average"] = False
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

    # super dictionary
    h5_save_dict = {}
    h5_save_dict["auto_corr"] = sav_dict["auto_corr"]
    h5_save_dict["auto_tile_i"] = sav_dict["auto_tile_i"]
    h5_save_dict = recarray_to_dict(h5_save_dict)

    save(after_file, h5_save_dict, "after_file")

    return after_file


def test_points_offzenith_zenith_1088716296(before_file, after_file):
    """Runs the test on `vis_extract_autocorr` - reads in the data in before_file and after_file,
    and then calls `vis_extract_autocorr`, checking the outputs match expectations"""
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    h5_after = load(after_file)

    obs = h5_before["obs"]
    vis_arr = h5_before["vis_arr"]
    pyfhd_config = h5_before["pyfhd_config"]

    expected_auto_corr = h5_after["auto_corr"]
    expected_auto_tile_i = h5_after["auto_tile_i"]

    result_auto_corr, result_auto_tile_i = vis_extract_autocorr(
        obs, vis_arr, pyfhd_config
    )

    # Outputs from .sav file can be an a 1D array containging 2D arrays
    # pyfhd is making 3D arrays. So test things match as a loop
    # over polarisations. Furthermore, FHD and pyfhd have different
    # axes orders, so transpose the expected values

    for pol_i in range(obs["n_pol"]):
        assert np.allclose(
            expected_auto_corr[pol_i].transpose(), result_auto_corr[pol_i], atol=1e-8
        )

    assert np.array_equal(result_auto_tile_i, expected_auto_tile_i)

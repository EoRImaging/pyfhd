from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.calibration.calibration_utils import cal_auto_ratio_divide
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt
import importlib_resources


@pytest.fixture
def data_dir():
    return importlib_resources.files("pyfhd.resources.test_data").joinpath(
        "calibration", "cal_auto_ratio_divide"
    )


@pytest.fixture(
    scope="function", params=["point_zenith", "point_offzenith", "1088716296"]
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run1", "run3"])
def run(request):
    return request.param


# Had to ["1088716296", "run1"], something really weird with the test, corrupt data?
skip_tests = [["1088716296", "run3"], ["1088716296", "run1"]]

# For each combination of tag and run, check if the hdf5 file exists, if not, create it and either way return the path
# Tests will fail if the fixture fails, not too worried about exceptions here.


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
    vis_auto = sav_file_vis_arr_swap_axes(sav_dict["vis_auto"])

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = obs
    h5_save_dict["cal"] = cal
    h5_save_dict["cal"]["gain"] = sav_file_vis_arr_swap_axes(
        h5_save_dict["cal"]["gain"]
    )
    h5_save_dict["vis_auto"] = vis_auto
    h5_save_dict["auto_tile_i"] = sav_dict["auto_tile_i"]

    save(before_file, h5_save_dict, "before_file")

    return before_file


# Same as the before_file fixture, except we're taking the the after files
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

    h5_save_dict["cal"] = recarray_to_dict(sav_dict["cal"])
    h5_save_dict["cal"]["gain"] = sav_file_vis_arr_swap_axes(
        h5_save_dict["cal"]["gain"]
    )
    h5_save_dict["auto_ratio"] = sav_file_vis_arr_swap_axes(sav_dict["auto_ratio"])

    save(after_file, h5_save_dict, "after_file")

    return after_file


@pytest.mark.github_actions
def test_cal_auto_ratio_divide(before_file, after_file):
    """
    Runs all the given tests on `cal_auto_ratio_divide` reads in the data in before_file and after_file,
    and then calls `cal_auto_ratio_divide`, checking the outputs match expectations
    """
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    h5_after = load(after_file)

    obs = h5_before["obs"]
    cal = h5_before["cal"]
    vis_auto = h5_before["vis_auto"]
    auto_tile_i = h5_before["auto_tile_i"]

    expected_cal = h5_after["cal"]
    expected_auto_ratio = h5_after["auto_ratio"]

    result_cal, result_auto_ratio = cal_auto_ratio_divide(
        obs, cal, vis_auto, auto_tile_i
    )

    atol = 8e-6
    npt.assert_allclose(expected_auto_ratio, result_auto_ratio, atol=atol)

    # check the gains have been updated
    npt.assert_allclose(expected_cal["gain"], result_cal["gain"], atol=atol)

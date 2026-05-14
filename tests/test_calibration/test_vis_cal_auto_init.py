from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.calibration.calibration_utils import vis_cal_auto_init
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_cal_auto_init")


@pytest.fixture(
    scope="function", params=["point_zenith", "point_offzenith", "1088716296"]
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run2", "run3"])
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

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = obs
    h5_save_dict["cal"] = cal
    h5_save_dict["vis_arr"] = sav_file_vis_arr_swap_axes(sav_dict["vis_arr"])
    h5_save_dict["vis_model_arr"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_model_arr"]
    )

    h5_save_dict["vis_auto"] = sav_file_vis_arr_swap_axes(sav_dict["vis_auto"])
    h5_save_dict["vis_model_auto"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_model_auto"]
    )

    h5_save_dict["auto_tile_i"] = sav_dict["auto_tile_i"]

    # Delete the ragged array
    del h5_save_dict["cal"]["mode_params"]

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
    h5_save_dict["auto_gain"] = sav_file_vis_arr_swap_axes(sav_dict["auto_gain"])

    save(after_file, h5_save_dict, "after_file")

    return after_file


def test_vis_cal_auto_init(before_file, after_file):
    """Runs the test on `vis_cal_auto_init` - reads in the data in before_file & after_file,
    and then calls `vis_cal_auto_init`, checking the outputs match expectations"""
    if before_file is None or after_file is None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    expected_auto_gain = load(after_file)

    obs = h5_before["obs"]
    cal = h5_before["cal"]
    vis_arr = h5_before["vis_arr"]
    vis_model_arr = h5_before["vis_model_arr"]
    vis_auto = h5_before["vis_auto"]
    vis_model_auto = h5_before["vis_model_auto"]
    auto_tile_i = h5_before["auto_tile_i"]

    result_auto_gain = vis_cal_auto_init(
        obs, cal, vis_arr, vis_model_arr, vis_auto, vis_model_auto, auto_tile_i
    )

    # Check returned gain is as expected
    # The atol is is a higher value than 1e-8 as resistant_mean has large enough
    # differences when using single vs double that it can cause issues upto
    # 1e-2 in some basic cases.
    atol = 4e-4
    if before_file.name.split("_")[0] == "1088716296":
        # This is one of those cases as the vis_auto_model array is just an array of ones
        # until a multiplication of auto_scale which it's values are different at the 2nd
        # decimal point again due to resistant_mean doing single precision by default unless
        # the double keyword is used, which in this case it was not.
        atol = 0.03
    npt.assert_allclose(result_auto_gain, expected_auto_gain, atol=atol)

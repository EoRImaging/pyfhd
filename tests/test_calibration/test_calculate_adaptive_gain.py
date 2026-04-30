import pytest
from os import environ as env
from pathlib import Path
from pyfhd.pyfhd_tools.test_utils import get_data_items
from pyfhd.calibration.calibration_utils import calculate_adaptive_gain
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt
import importlib_resources


@pytest.fixture
def data_dir():
    return importlib_resources.files("pyfhd.resources.test_data").joinpath(
        "calibration", "calculate_adaptive_gain"
    )


@pytest.fixture(scope="function", params=["point_zenith", "point_offzenith"])
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3"])
def run(request):
    return request.param


skip_tests = []


@pytest.fixture
def before_file(data_dir, tag, run):
    if [tag, run] in skip_tests:
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")

    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    # The dict happens to be good already
    save(before_file, sav_dict, "before_file")

    return before_file


@pytest.fixture
def after_file(data_dir, tag, run):
    if [tag, run] in skip_tests:
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    # After is also good in this case
    save(after_file, sav_dict, "after_file")

    return after_file


@pytest.fixture(scope="function", params=[1, 2])
def calc_test(request):
    return request.param


@pytest.fixture
def calc_test_before(data_dir, calc_test):
    before_file = Path(data_dir, f"test_{calc_test}_before_{data_dir.name}.h5")

    if before_file.exists():
        return before_file

    gain_list, convergence_list, iter, base_gain, final_con_est = get_data_items(
        data_dir,
        f"input_gain_list_{calc_test}.npy",
        f"input_convergence_list_{calc_test}.npy",
        f"input_iter_{calc_test}.npy",
        f"input_base_gain_{calc_test}.npy",
        f"input_final_convergence_estimate_{calc_test}.npy",
    )

    h5_save_dict = {}
    h5_save_dict["gain_list"] = gain_list
    h5_save_dict["convergence_list"] = convergence_list
    h5_save_dict["iter"] = iter
    h5_save_dict["base_gain"] = base_gain
    h5_save_dict["final_convergence_estimate"] = final_con_est

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture
def calc_test_after(data_dir, calc_test):
    after_file = Path(data_dir, f"test_{calc_test}_after_{data_dir.name}.h5")

    if after_file.exists():
        return after_file

    expected_gain = get_data_items(
        data_dir,
        f"output_gain_{calc_test}.npy",
    )

    h5_save_dict = {}
    h5_save_dict["expected_gain"] = expected_gain

    save(after_file, h5_save_dict, "after_file")

    return after_file


@pytest.mark.github_actions
def test_point_offzenith_and_zenith(before_file, after_file):
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    h5_after = load(after_file)

    result_gain = calculate_adaptive_gain(
        h5_before["gain_list"],
        h5_before["convergence_list"],
        h5_before["iter"],
        h5_before["base_gain"],
        h5_before["final_convergence_estimate"],
    )

    npt.assert_almost_equal(h5_after["gain"], result_gain)


@pytest.mark.github_actions
def test_calc_test_1_and_2(calc_test_before, calc_test_after):

    h5_before = load(calc_test_before)
    expected_gain = load(calc_test_after)

    result_gain = calculate_adaptive_gain(
        h5_before["gain_list"],
        h5_before["convergence_list"],
        h5_before["iter"],
        h5_before["base_gain"],
        h5_before["final_convergence_estimate"],
    )

    npt.assert_almost_equal(expected_gain, result_gain)


"""
The below test will never pass due to IDL's default median behaviour, for example
PRINT, MEDIAN([1, 2, 3, 4], /EVEN) 
2.50000
PRINT, MEDIAN([1, 2, 3, 4])
3.00000

One has to use the EVEN keyword to get the *proper* median behaviour that most mathematicians and scientists
expect. This was the only thing that produced the wrong results was the use of numpy median instead of IDL MEDIAN.
The array that went into it est_final_conv produced the exact same results in Python and IDL (with the exception of
Python producing a more accurate result due to double precision) 

def test_calc_adapt_gain_three(data_dir):
    gain_list, convergence_list, iter, base_gain, expected_gain = get_data_items(
        data_dir,
        'input_gain_list_3.npy',
        'input_convergence_list_3.npy',
        'input_iter_3.npy',
        'input_base_gain_3.npy',
        'output_gain_3.npy'
    )
    result_gain = calculate_adaptive_gain(gain_list, convergence_list, iter, base_gain)
    assert expected_gain == result_gain
"""

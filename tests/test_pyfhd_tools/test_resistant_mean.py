from os import environ as env
from pathlib import Path

import numpy as np
import pytest
from numpy import testing as npt

from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, save
from pyfhd.pyfhd_tools.pyfhd_utils import resistant_mean


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "pyfhd_tools", "resistant_mean")


@pytest.fixture(
    scope="function", params=["point_zenith", "point_offzenith", "1088716296"]
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run2", "run3"])
def run(request):
    return request.param


# Last 3 are due to resistant_mean calculating in single precision in IDL unless
# double keyword is used
skip_tests = [
    ["1088716296", "run3"],
    ["1088716296", "run2"],
    ["point_zenith", "run2"],
    ["point_zenith", "run3"],
]


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

    input_array = sav_dict["input_array"]
    deviations = sav_dict["deviations"]

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["input_array"] = input_array
    h5_save_dict["deviations"] = deviations

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

    res_mean_data = sav_dict["res_mean_data"]

    save(after_file, np.array([res_mean_data], dtype=np.float64), "after_file")

    return after_file


def test_points_zenith_offzenith_and_1088716296(before_file, after_file):
    """
    Runs the test on `resistant_mean`. It reads in the data in before_file and
    after_file, and then calls `resistant_mean`, checking the outputs match expectations
    """
    if before_file is None or after_file is None:
        pytest.skip(
            "This test has been skipped. In this case precision played a major "
            "factor, resistant_mean when using the double keyword in IDL will "
            "get the same result as Python, but the tests taken here were "
            f"single_precision. The skipped tests are: {skip_tests}"
        )

    h5_before = load(before_file)
    expected_res_mean = load(after_file)

    input_array = h5_before["input_array"]
    deviations = h5_before["deviations"]

    result_res_mean = resistant_mean(input_array, deviations)

    assert np.allclose(result_res_mean, expected_res_mean, atol=1e-4)


@pytest.mark.github_actions
def test_res_mean_int():
    input_array = np.concatenate([np.arange(20), np.array([100, 200, 300, 400])])
    assert resistant_mean(input_array, 2) == 9.5


@pytest.mark.github_actions
def test_res_mean_float():
    input_array = np.concatenate(
        [np.arange(0, 20, 0.75), np.array([25.0, -10.75, -30.0, 50])]
    )
    assert resistant_mean(input_array, 2) == 9.75


@pytest.mark.github_actions
def test_res_mean_complex_int():
    input_array = np.linspace(0 + 2j, 20 + 42j, 21)
    npt.assert_allclose(resistant_mean(input_array, 2), 7 + 16j)


@pytest.mark.github_actions
def test_res_mean_complex_float():
    input_array = np.linspace(0 + 10j, 10 + 30j, 20)
    npt.assert_allclose(
        resistant_mean(input_array, 3), 2.6315789872949775 + 15.263158017938787j
    )


@pytest.mark.github_actions
def test_res_mean_complex_large_i():
    input_array = np.concatenate(
        [np.linspace(0, 19 + 19j, 20), np.array([1 + 100j, 3 + 400j, 5 + 500j])]
    )
    npt.assert_allclose(resistant_mean(input_array, 3), 9.5 + 9.5j)


@pytest.mark.github_actions
def test_res_mean_random_large():
    input_array = np.concatenate(
        [np.linspace(0, 10, 100_000), np.arange(-1_000_000, 1_000_000, 1000)]
    )
    npt.assert_allclose(resistant_mean(input_array, 4), 4.9998998746918923, atol=1e-4)

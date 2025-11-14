import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.pyfhd_tools.pyfhd_utils import weight_invert
import importlib_resources


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "pyfhd_tools", "weight_invert")


@pytest.mark.github_actions
def test_weight_invert_one():
    data_dir = importlib_resources.files("PyFHD.resources.test_data").joinpath(
        "pyfhd_tools", "weight_invert"
    )
    threshold, weights, expected_result = get_data_items(
        data_dir,
        "visibility_grid_input_threshold_1.npy",
        "visibility_grid_input_weights_1.npy",
        "visibility_grid_output_result_1.npy",
    )
    result = weight_invert(weights, threshold=threshold)
    assert np.array_equal(result, expected_result)


def test_weight_invert_two(data_dir):
    threshold, weights, expected_result = get_data_items(
        data_dir,
        "visibility_grid_input_threshold_2.npy",
        "visibility_grid_input_weights_2.npy",
        "visibility_grid_output_result_2.npy",
    )
    result = weight_invert(weights, threshold=threshold)
    np.testing.assert_allclose(result, expected_result, rtol=0, atol=1e-11)


def test_weight_invert_three(data_dir):
    threshold, weights, expected_result, abs = get_data_items(
        data_dir,
        "visibility_grid_input_threshold_3.npy",
        "visibility_grid_input_weights_3.npy",
        "visibility_grid_output_result_3.npy",
        "visibility_grid_input_abs_3.npy",
    )
    result = weight_invert(weights, threshold=threshold, use_abs=abs)
    np.testing.assert_allclose(result, expected_result, rtol=0, atol=1e-11)


def test_weight_invert_four(data_dir):
    weights, expected_result = get_data_items(
        data_dir,
        "visibility_grid_input_weights_4.npy",
        "visibility_grid_output_result_4.npy",
    )
    result = weight_invert(weights)
    assert np.array_equal(result, expected_result)


def test_weight_invert_five(data_dir):
    weights, expected_result = get_data_sav(data_dir, "input_5.sav", "output_5.sav")
    result = weight_invert(weights)
    assert np.array_equal(result, expected_result)

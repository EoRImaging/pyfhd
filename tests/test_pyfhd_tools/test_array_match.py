import numpy as np
from pyfhd.pyfhd_tools.test_utils import get_data_items
from pyfhd.pyfhd_tools.pyfhd_utils import array_match
import pytest
import importlib_resources


@pytest.fixture
def data_dir():
    return importlib_resources.files("pyfhd.resources.test_data").joinpath(
        "pyfhd_tools", "array_match"
    )


@pytest.mark.github_actions
def test_array_match_1(data_dir):
    array1, array2, value_match, expected_indices = get_data_items(
        data_dir,
        "vis_weights_update_input_array1.npy",
        "vis_weights_update_input_array2.npy",
        "vis_weights_update_input_value_match.npy",
        "vis_weights_update_output_match_indices.npy",
    )

    # Get the result and see if they match.
    indices = array_match(array1, value_match, array_2=array2)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_array_match_2(data_dir):
    array1, value_match, expected_indices = get_data_items(
        data_dir,
        "input_array1_2.npy",
        "input_value_match_2.npy",
        "output_match_indices_2.npy",
    )
    # Get the result and see if they match.
    indices = array_match(array1, value_match)
    print(indices.shape)
    print(expected_indices.shape)
    print(indices - expected_indices)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_array_match_3(data_dir):
    array1, array2, value_match, expected_indices = get_data_items(
        data_dir,
        "input_array1_3.npy",
        "input_array2_3.npy",
        "input_value_match_3.npy",
        "output_match_indices_3.npy",
    )
    # Get the result and see if they match.
    indices = array_match(array1, value_match, array_2=array2)
    assert np.array_equal(indices, expected_indices)

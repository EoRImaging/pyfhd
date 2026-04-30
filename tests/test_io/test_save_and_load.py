from pyfhd.io.pyfhd_io import load, save
from pathlib import Path
import numpy as np
from h5py import File, Group, Dataset
import pytest


@pytest.mark.github_actions
def test_save_and_load():
    """
    Test the save and load functionality of pyfhd.
    This function checks if the data can be saved to a file and then loaded back correctly.
    """
    # Create a sample dictionary to save
    sample_data = {
        "key1": [1, 2, 3],
        "key2": {"subkey1": "value1", "subkey2": "value2"},
        "key3": 42,
        "key4": None,
    }

    # Save the sample data to a file
    save("test_data.h5", sample_data, "sample")

    # Load the data back from the file
    loaded_data = load("test_data.h5")

    # Check if the loaded data matches the original sample data
    for key in sample_data.keys():
        assert key in loaded_data, f"Key {key} not found in loaded data"
        if isinstance(sample_data[key], dict):
            # For dictionaries, check if all keys match
            for subkey in sample_data[key].keys():
                assert (
                    subkey in loaded_data[key]
                ), f"Subkey {subkey} not found in loaded data[{key}]"
                assert (
                    loaded_data[key][subkey] == sample_data[key][subkey]
                ), f"Value for subkey {subkey} does not match in loaded data[{key}]"
        elif isinstance(sample_data[key], list):
            # For lists, check if they match
            assert np.array_equal(
                loaded_data[key], sample_data[key]
            ), f"List for key {key} does not match"
        else:
            assert (
                loaded_data[key] == sample_data[key]
            ), f"Value for key {key} does not match"

    Path("test_data.h5").unlink()  # Clean up the test file


@pytest.mark.github_actions
def test_save_and_load_empty():
    """
    Test the save and load functionality with an empty dictionary.
    This function checks if an empty dictionary can be saved and loaded correctly.
    """
    # Create an empty dictionary to save
    empty_data = {}

    # Save the empty data to a file"
    save("empty_data.h5", empty_data, "empty")

    # Load the data back from the file
    loaded_empty_data = load("empty_data.h5")

    # Check if the loaded data is still an empty dictionary
    assert loaded_empty_data == empty_data

    Path("empty_data.h5").unlink()  # Clean up the test file


@pytest.mark.github_actions
def test_lazy_load():
    """
    Test the lazy loading functionality of pyfhd.
    This function checks if the data can be loaded lazily and accessed correctly.
    """
    # Create a sample dictionary to save
    sample_data = {
        "key1": [1, 2, 3],
        "key2": {"subkey1": "value1", "subkey2": "value2"},
        "key3": 42,
        "key4": None,
    }

    # Save the sample data to a file
    save("lazy_data.h5", sample_data, "lazy_sample")

    # Load the data lazily
    lazy_loaded_data = load("lazy_data.h5", lazy_load=True)

    assert isinstance(
        lazy_loaded_data, File
    ), "Lazy loaded data is not an h5py File object"

    assert isinstance(lazy_loaded_data["key1"], Dataset)
    assert np.array_equal(
        lazy_loaded_data["key1"][:], sample_data["key1"]
    ), "Lazy loaded data for key1 does not match"

    assert isinstance(
        lazy_loaded_data["key2"], Group
    ), "Lazy loaded data does not contain the expected group"

    Path("lazy_data.h5").unlink()  # Clean up the test file

from h5py import File, Group, Dataset
from scipy.sparse import csr_array
from pyuvdata import AiryBeam
import numpy as np
import pytest

from pyfhd.io.pyfhd_io import load, save


@pytest.mark.github_actions
def test_save_and_load_dict(tmp_path):
    """
    Test the save and load functionality of pyfhd.
    This function checks if the data can be saved to a file and then loaded back
    correctly.
    """
    # Create a sample dictionary to save
    sample_data = {
        "key1": [1, 2, 3],
        "key2": {"subkey1": "value1", "subkey2": {"foo": 3, "bar": 4}},
        "key3": 42,
        "key4": None,
        "key5": np.array([0, 0, 1, 2, 2, 2]),
        "key6": csr_array(
            (
                np.array([1, 2, 3, 4, 5, 6]),
                (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2])),
            ),
            shape=(3, 3),
        ),
        "key7": AiryBeam(diameter=14),
    }

    file_path = tmp_path / "test_data.h5"
    # Save the sample data to a file
    save(file_path, sample_data, "sample")

    # Load the data back from the file
    loaded_data = load(file_path)

    # Check if the loaded data matches the original sample data
    for key in sample_data.keys():
        assert key in loaded_data, f"Key {key} not found in loaded data"
        if isinstance(sample_data[key], dict):
            # For dictionaries, check if all keys match
            for subkey in sample_data[key].keys():
                assert subkey in loaded_data[key], (
                    f"Subkey {subkey} not found in loaded data[{key}]"
                )
                assert loaded_data[key][subkey] == sample_data[key][subkey], (
                    f"Value for subkey {subkey} does not match in loaded data[{key}]"
                )
        elif isinstance(sample_data[key], (list, np.ndarray)):
            # For lists, check if they match
            assert np.array_equal(loaded_data[key], sample_data[key]), (
                f"List for key {key} does not match"
            )
        elif isinstance(sample_data[key], csr_array):
            assert (
                loaded_data[key].shape == sample_data[key].shape
                and np.all(loaded_data[key].indices == sample_data[key].indices)
                and np.all(loaded_data[key].indptr == sample_data[key].indptr)
                and np.allclose(loaded_data[key].data, sample_data[key].data)
            )
        else:
            assert loaded_data[key] == sample_data[key], (
                f"Value for key {key} does not match"
            )


@pytest.mark.github_actions
@pytest.mark.parametrize(
    "sample_data",
    [
        np.array([0, 0, 1, 2, 2, 2]),
        csr_array(
            (
                np.array([1, 2, 3, 4, 5, 6]),
                (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2])),
            ),
            shape=(3, 3),
        ),
        [0, 0, 1, 2, 2, 2],
        3.14,
    ],
)
def test_save_and_load_single(tmp_path, sample_data):
    """
    Test the save and load functionality of pyfhd.
    This function checks if the data can be saved to a file and then loaded back
    correctly.
    """
    file_path = tmp_path / "test_data.h5"
    # Save the sample data to a file
    save(file_path, sample_data, "sample")

    # Load the data back from the file
    loaded_data = load(file_path)

    # Check if the loaded data matches the original sample data
    if isinstance(sample_data, (list, np.ndarray)):
        # For lists, check if they match
        assert np.array_equal(loaded_data, sample_data)
    elif isinstance(sample_data, csr_array):
        assert (
            loaded_data.shape == sample_data.shape
            and np.all(loaded_data.indices == sample_data.indices)
            and np.all(loaded_data.indptr == sample_data.indptr)
            and np.allclose(loaded_data.data, sample_data.data)
        )
    else:
        assert loaded_data == sample_data


@pytest.mark.github_actions
def test_save_and_load_sparse_error(tmp_path):
    sample_data = {
        "key1": {
            "sparse_array_type": "coo_array",
            "data": np.array([1, 2, 3, 4, 5, 6]),
            "row": np.array([0, 0, 1, 2, 2, 2]),
            "column": np.array([0, 2, 2, 0, 1, 2]),
            "shape": (3, 3),
        }
    }
    file_path = tmp_path / "test_data.h5"
    # Save the sample data to a file
    save(file_path, sample_data, "sample")

    # Load the data back from the file
    with pytest.raises(
        NotImplementedError,
        match="coo_array sparse array type detected, load only supports csr "
        "sparse arrays currently.",
    ):
        load(file_path)


@pytest.mark.github_actions
def test_save_and_load_empty(tmp_path):
    """
    Test the save and load functionality with an empty dictionary.
    This function checks if an empty dictionary can be saved and loaded correctly.
    """
    # Create an empty dictionary to save
    empty_data = {}

    # Save the empty data to a file"
    file_path = tmp_path / "empty_data.h5"
    save(file_path, empty_data, "empty")

    # Load the data back from the file
    loaded_empty_data = load(file_path)

    # Check if the loaded data is still an empty dictionary
    assert loaded_empty_data == empty_data


@pytest.mark.github_actions
def test_load_file_without_empty_sentinel_attribute(tmp_path):
    """Files not written by pyfhd's save() (e.g. HDF5 produced by deepdish or
    PyTables, like several of the bundled healpix inds resources) do not carry
    the per-dataset "is empty" sentinel attribute. load() must read such a
    dataset as real data instead of raising a KeyError for the missing
    attribute.
    """
    expected = np.arange(10, dtype=np.int32)

    # Mimic a foreign file: a real dataset plus unrelated metadata attributes,
    # but no matching "hpx_inds" sentinel attribute.
    file_path = tmp_path / "foreign_data.h5"
    with File(file_path, "w") as f:
        f.create_dataset("hpx_inds", data=expected)
        f.attrs["nside"] = 512

    loaded_data = load(file_path)

    assert np.array_equal(loaded_data, expected), (
        "Dataset without a sentinel attribute was not loaded correctly"
    )


@pytest.mark.github_actions
def test_lazy_load(tmp_path):
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
    file_path = tmp_path / "lazy_data.h5"
    save(file_path, sample_data, "lazy_sample")

    # Load the data lazily
    lazy_loaded_data = load(file_path, lazy_load=True)

    assert isinstance(lazy_loaded_data, File), (
        "Lazy loaded data is not an h5py File object"
    )

    assert isinstance(lazy_loaded_data["key1"], Dataset)
    assert np.array_equal(lazy_loaded_data["key1"][:], sample_data["key1"]), (
        "Lazy loaded data for key1 does not match"
    )

    assert isinstance(lazy_loaded_data["key2"], Group), (
        "Lazy loaded data does not contain the expected group"
    )

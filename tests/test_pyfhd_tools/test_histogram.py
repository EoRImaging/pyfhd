import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from pyfhd.pyfhd_tools.pyfhd_utils import histogram
from pyfhd.pyfhd_tools.test_utils import get_data, get_data_items
import importlib_resources


@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return importlib_resources.files("pyfhd.resources.test_data").joinpath(
        "pyfhd_tools", "histogram"
    )


@pytest.fixture
def full_data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "pyfhd_tools", "histogram")


@pytest.mark.github_actions
def test_idl_example(data_dir: Path):
    """
    This test is based on the example from the IDL documentation.
    This ensures we get the same behaviour as an example everybody can see.
    """
    # Setup the test from the histogram data file
    data, expected_hist, expected_indices = get_data(
        data_dir, "idl_hist_example.npy", "idl_example_hist.npy", "idl_example_inds.npy"
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_one_hundred_nums(data_dir: Path):
    """
    This is a basic test of an array with numbers 0 to 99 in increasing
    order.
    Should produce two bins.
    """
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "hundred_ints.npy",
        "hundred_ints_hist_bin50.npy",
        "hundred_ints_inds_bin50.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    print(data)
    hist, _, indices = histogram(data, bin_size=50)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_one_hundred_ten_bins(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "hundred_ints.npy",
        "hundred_ints_hist_nbin10.npy",
        "hundred_ints_inds_nbin10.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    # This is to show that bin_size is ignored when num_bins is used
    hist, _, indices = histogram(data, num_bins=10, bin_size=1000)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_min(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "hundred_ints.npy",
        "hundred_ints_hist_min10.npy",
        "hundred_ints_inds_min10.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, bin_size=10, min=10)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_ones_and_min():
    only_ones = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    output, _, ri = histogram(only_ones, min=1)
    expected_hist = np.array([3])
    expected_ri = np.array([2, 5, 4, 5, 6])
    assert np.array_equal(output, expected_hist)
    assert np.array_equal(ri, expected_ri)


@pytest.mark.github_actions
def test_max(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "hundred_ints.npy",
        "hundred_ints_hist_max50.npy",
        "hundred_ints_inds_max50.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, bin_size=10, max=50)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_min_max(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "hundred_ints.npy",
        "hundred_ints_hist_min10_max55.npy",
        "hundred_ints_inds_min10_max55.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, bin_size=10, min=10, max=55)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_find_none():
    only_twos = np.array([0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0])
    output, _, ri = histogram(only_twos, min=1, max=1)
    expected_hist = np.array([0])
    expected_ri = np.array([2, 2])
    assert np.array_equal(output, expected_hist)
    assert np.array_equal(ri, expected_ri)


@pytest.mark.github_actions
def test_one_max(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "hundred_ints.npy",
        "hundred_ints_hist_binsize1_max55.npy",
        "hundred_ints_inds_binsize1_max55.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, max=55)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_normals(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir, "normals.npy", "normals_hist.npy", "normals_inds.npy"
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_normals_binsize(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "normals.npy",
        "normals_hist_binsize025.npy",
        "normals_inds_binsize025.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, bin_size=0.25)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_normals_min_max(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir,
        "normals.npy",
        "normals_hist_min_max.npy",
        "normals_inds_binsize_min_max.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, min=0, max=1, bin_size=0.25)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


@pytest.mark.github_actions
def test_normals_times_10(data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        data_dir, "normals.npy", "normals_hist_times10.npy", "normals_inds_times10.npy"
    )
    data = data * 10
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, bin_size=2)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


def test_billion(full_data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        full_data_dir, "one_billion.npy", "one_billion_hist.npy", "one_billion_inds.npy"
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


def test_billion_floats(full_data_dir: Path):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(
        full_data_dir,
        "one_billion_floats.npy",
        "one_billion_floats_hist.npy",
        "one_billion_floats_inds.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)


def test_full_size(full_data_dir: Path):
    # Read the histogram file
    data, binsize, min, expected_hist, expected_indices = get_data_items(
        full_data_dir,
        "input_1.npy",
        "binsize_1.npy",
        "min_1.npy",
        "output_1.npy",
        "reverse_indices_1.npy",
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, bin_size=binsize, min=min)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

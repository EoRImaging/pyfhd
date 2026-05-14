from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
import h5py
from logging import Logger
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.healpix.healpix_utils import healpix_cnv_apply
import importlib_resources


@pytest.fixture
def data_dir():
    if env.get("PYFHD_TEST_PATH"):
        return Path(env.get("PYFHD_TEST_PATH"), "healpix", "healpix_cnv_apply")
    else:
        return None


@pytest.fixture(
    scope="function",
    params=[
        "1088285600",
        "1088716296",
        pytest.param("point_zenith", marks=pytest.mark.github_actions),
        "point_offzenith",
    ],
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3", "run4"])
def run(request):
    return request.param


@pytest.fixture(
    scope="function", params=["", "beam", "dirty", "model", "variance", "weights"]
)
def array_type(request):
    return request.param


skip_tests = [
    ["1088285600", "run4"],
    ["point_zenith", "run3"],
    ["point_offzenith", "run3"],
    ["1088716296", "run3"],
]


@pytest.fixture
def before_file(tag, run, array_type, data_dir, request):
    if [tag, run] in skip_tests:
        return None
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "healpix", "healpix_cnv_apply"
        )
    type_to_add = "" if array_type == "" else f"_{array_type}"
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}{type_to_add}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)
    # This will allow the arrays to be saved a variable lengths in the before file
    sav_dict["hpx_cnv"]["ija"] = np.array(sav_dict["hpx_cnv"]["ija"], dtype=object)
    sav_dict["hpx_cnv"]["sa"] = np.array(sav_dict["hpx_cnv"]["sa"], dtype=object)
    sav_dict["hpx_cnv"]["i_use"] = sav_dict["hpx_cnv"]["i_use"].astype(np.int64)

    h5_save = {
        "hpx_cnv": sav_dict["hpx_cnv"],
        "image": sav_dict["image"],
    }

    variable_lengths = {
        "ija": h5py.vlen_dtype(np.int64),
        "sa": h5py.vlen_dtype(np.float64),
    }

    save(before_file, h5_save, "sav_dict", variable_lengths=variable_lengths)

    return before_file


@pytest.fixture()
def after_file(tag, run, array_type, data_dir, request):
    if [tag, run] in skip_tests:
        return None
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "healpix", "healpix_cnv_apply"
        )
    type_to_add = "" if array_type == "" else f"_{array_type}"
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}{type_to_add}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    save(after_file, sav_dict["hpx_map"], "hpx_map")

    return after_file


def test_healpix_cnv_apply(before_file, after_file, request):
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )
    # This was done here to make it work in GitHub Actions
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "healpix", "healpix_cnv_apply"
        )
        before_file = Path(data_dir, before_file.name)
        after_file = Path(data_dir, after_file.name)

    h5_before = load(before_file)
    expected_hpx_map = load(after_file)

    hpx_map = healpix_cnv_apply(h5_before["image"], h5_before["hpx_cnv"])

    # Have to test this differently because we're dealing with numbers
    # of differing magintudes, So we're using the rtol and atol together rather
    # than just the atol
    npt.assert_allclose(hpx_map, expected_hpx_map, rtol=1e-5, atol=0.001)

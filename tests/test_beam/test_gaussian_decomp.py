from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from logging import Logger
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.beam_setup.beam_utils import gaussian_decomp


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "beam_setup", "gaussian_decomp")


@pytest.fixture(
    scope="function",
    params=["1088285600", "1088716296", "point_zenith", "point_offzenith"],
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3", "run4"])
def run(request):
    return request.param


skip_tests = [
    ["1088285600", "run4"],
    ["point_zenith", "run3"],
    ["point_offzenith", "run3"],
    ["1088716296", "run3"],
]


@pytest.fixture
def before_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    del sav_dict["after_filename"]
    del sav_dict["before_filename"]
    del sav_dict["file_path_fhd"]
    del sav_dict["extra"]

    save(before_file, sav_dict, "gaussian_decomp")

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
    sav_dict = recarray_to_dict(sav_dict)

    save(after_file, sav_dict["decomp_beam"], "decomp_beam")

    return after_file


def test_gaussian_decomp(before_file, after_file):
    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    expected_decomp_beam = load(after_file)

    decomp_beam, _, _ = gaussian_decomp(
        h5_before["x"],
        h5_before["y"],
        h5_before["p"],
        model_npix=h5_before["model_npix"],
        model_res=h5_before["model_res"],
    )

    npt.assert_allclose(decomp_beam, expected_decomp_beam, atol=1e-16)

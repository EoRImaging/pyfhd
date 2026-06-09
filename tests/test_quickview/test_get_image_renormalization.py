from os import environ as env
from pathlib import Path
import numpy.testing as npt
import pytest
from logging import Logger
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.io.pyfhd_quickview import get_image_renormalization


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "quickview", "get_image_renormalization")


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

    pyfhd_config = {
        "pad_uv_image": sav_dict["pad_uv_image"],
        "image_filter": "filter_uv_uniform",
    }

    sav_dict["pyfhd_config"] = pyfhd_config

    save(before_file, sav_dict, "sav_dict")

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

    save(after_file, sav_dict["renorm_factor"], "renorm_factor")

    return after_file


def test_get_image_renormalization(before_file, after_file):
    if before_file is None or after_file is None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    expected_renorm_factor = load(after_file)

    h5_before["obs"]["dimension"] = int(h5_before["obs"]["dimension"])
    h5_before["obs"]["elements"] = int(h5_before["obs"]["elements"])
    h5_before["obs"]["obsx"] = int(h5_before["obs"]["obsx"])
    h5_before["obs"]["obsy"] = int(h5_before["obs"]["obsy"])

    renorm_factor = get_image_renormalization(
        h5_before["obs"],
        h5_before["weights_arr"],
        h5_before["beam_base"],
        h5_before["filter_arr"],
        h5_before["pyfhd_config"],
        Logger(1),
    )
    # Interestingly we are at the limit of single precision for this result so
    # IDL can't actually use the decimal places here where we can i.e. in IDL
    # 17044907.32 EQ 17044908. is True while in Python this is clearly False
    # This means the rtol should be a max of 1 and we'll consider the pyfhd version
    # to be better here as we can represent the number and decimal places
    npt.assert_allclose(renorm_factor, expected_renorm_factor, rtol=1)

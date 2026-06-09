from os import environ as env
from pathlib import Path
import numpy.testing as npt
import pytest
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.beam_setup.beam_utils import beam_image


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "beam_setup", "beam_image")


@pytest.fixture
def beam_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "beams")


@pytest.fixture(
    scope="function",
    params=["1088285600", "1088716296", "point_zenith", "point_offzenith"],
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3", "run4"])
def run(request):
    return request.param


@pytest.fixture(scope="function", params=["", "quickview"])
def quickview(request):
    return request.param


skip_tests = [
    ["1088285600", "run4"],
    ["point_zenith", "run3"],
    ["point_offzenith", "run3"],
    ["1088716296", "run3"],
]


@pytest.fixture
def before_file(tag, run, quickview, data_dir):
    if [tag, run] in skip_tests:
        return None
    if quickview != "":
        before_file = Path(
            data_dir, f"{tag}_{run}_before_{data_dir.name}_{quickview}.h5"
        )
    else:
        before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    save(before_file, sav_dict, "beam_image")

    return before_file


@pytest.fixture()
def after_file(tag, run, quickview, data_dir):
    if [tag, run] in skip_tests:
        return None
    if quickview != "":
        after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}_{quickview}.h5")
    else:
        after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    if quickview != "":
        save(after_file, sav_dict["beam_pol_0"], "beam_pol_0")
    else:
        save(after_file, sav_dict["beam_base"], "beam_base")

    return after_file


def test_beam_image(before_file, after_file, beam_dir):
    if before_file is None or after_file is None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    expected_beam_base = load(after_file)

    psf = load(Path(beam_dir, "decomp_beam_pointing0.h5"), lazy_load=True)

    h5_before["obs"]["dimension"] = int(h5_before["obs"]["dimension"])

    beam_base = beam_image(
        psf,
        h5_before["obs"],
        h5_before["pol_i"],
        freq_i=h5_before["freq_i"] if "freq_i" in h5_before else None,
        square=h5_before["square"] if "square" in h5_before else False,
    )

    npt.assert_allclose(beam_base, expected_beam_base, atol=1e-8)

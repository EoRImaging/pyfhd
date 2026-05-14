from os import environ as env
from pathlib import Path
import numpy.testing as npt
import pytest
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.healpix.healpix_utils import phase_shift_uv_image


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "healpix", "phase_shift_uv_image")


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

    obs = recarray_to_dict(sav_dict["obs"])

    save(before_file, obs, "obs")

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

    save(after_file, sav_dict["rephase_calc"], "rephase_expected")

    return after_file


def test_phase_shift_uv_image(before_file, after_file):
    if before_file is None or after_file is None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    obs = load(before_file)
    expected_rephase = load(after_file)

    obs["dimension"] = int(obs["dimension"])
    obs["elements"] = int(obs["elements"])

    rephase = phase_shift_uv_image(obs)

    # Precision differences caused by radec_to_pixel double precision calculation
    # vs single. Furthermore, the !Pi used in phase_shift_uv_image is also single
    # precision rather than the double precision variant IDL has. The calculations
    # are mathematically the same in both pyfhd and FHD. phase_shift_uv_image
    # apply_astrometry is also taking into account refraction.
    npt.assert_allclose(rephase, expected_rephase, atol=6e-5)

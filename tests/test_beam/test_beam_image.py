from os import environ as env
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.pyfhd_tools.test_utils import get_savs
from pyfhd.beam_setup.beam_utils import beam_image
from pyfhd.data.datasets import fetch_data


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
            "This test has been skipped, likely because we don't have the "
            f"required FHD output. It was listed in the skipped tests: {skip_tests}"
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


@pytest.mark.github_actions
@pytest.mark.parametrize(
    ("square", "use_abs"), [(False, False), (True, False), (True, True)]
)
def test_beam_image_psf_cut(zenith_obs_2013, zenith_psf_2013_cut, square, use_abs):
    psf = zenith_psf_2013_cut
    obs = zenith_obs_2013

    expected_beam_file = fetch_data("2013_zenith_beam_image")

    beam_image_dict = get_savs(expected_beam_file, "")
    beam_image_dict = recarray_to_dict(beam_image_dict)
    expected_beam = beam_image_dict["beam_image_arr"].T

    dimension = 256
    for pol_i in [0, 1]:
        beam_out = beam_image(
            psf,
            obs,
            pol_i,
            freq_i=0,
            dimension=dimension,
            square=square,
            use_abs=use_abs,
        )

        if square:
            # just do the calculation
            low_ind = int(dimension / 2 - psf["dim"] / 2 + 1)
            high_ind = int(dimension / 2 - psf["dim"] / 2 + psf["dim"])
            beam_base_uv = np.zeros([dimension, dimension], np.complex128)
            beam_single = (
                psf["beam_ptr"][pol_i, 0].reshape([psf["dim"], psf["dim"]])
            ).astype(np.complex128)
            if use_abs:
                beam_single = np.abs(beam_single)
            beam_base_uv[low_ind : high_ind + 1, low_ind : high_ind + 1] = beam_single
            beam_base = np.fft.fftshift(
                np.fft.ifftn(np.fft.fftshift(beam_base_uv), norm="forward")
            )
            comp_beam = (beam_base * np.conjugate(beam_base)).real
        else:
            comp_beam = expected_beam[pol_i]
        np.testing.assert_allclose(beam_out, comp_beam, rtol=0, atol=1e-9)

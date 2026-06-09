from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
import h5py
from logging import Logger
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.healpix.healpix_utils import healpix_cnv_generate
import importlib_resources


@pytest.fixture
def data_dir():
    if env.get("PYFHD_TEST_PATH"):
        return Path(env.get("PYFHD_TEST_PATH"), "healpix", "healpix_cnv_generate")
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


skip_tests = [
    ["1088285600", "run4"],
    ["point_zenith", "run3"],
    ["point_offzenith", "run3"],
    ["1088716296", "run3"],
]


@pytest.fixture
def before_file(tag, run, data_dir, request):
    if [tag, run] in skip_tests:
        return None
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "healpix", "healpix_cnv_generate"
        )
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    sav_dict["obs"]["dimension"] = int(sav_dict["obs"]["dimension"])
    sav_dict["obs"]["elements"] = int(sav_dict["obs"]["elements"])

    sav_dict["pyfhd_config"] = {
        "restrict_healpix_inds": sav_dict["restrict_hpx_inds"],
        "healpix_inds": None,
    }

    save(before_file, sav_dict, "sav_dict")

    return before_file


@pytest.fixture()
def after_file(tag, run, data_dir, request):
    if [tag, run] in skip_tests:
        return None
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "healpix", "healpix_cnv_generate"
        )
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    # This will allow the arrays to be saved a variable lengths in the before file
    sav_dict["hpx_cnv"]["ija"] = np.array(sav_dict["hpx_cnv"]["ija"], dtype=object)
    sav_dict["hpx_cnv"]["sa"] = np.array(sav_dict["hpx_cnv"]["sa"], dtype=object)
    sav_dict["hpx_cnv"]["i_use"] = sav_dict["hpx_cnv"]["i_use"].astype(np.int64)

    variable_lengths = {
        "ija": h5py.vlen_dtype(np.int64),
        "sa": h5py.vlen_dtype(np.float64),
    }

    save(after_file, sav_dict["hpx_cnv"], "hpx_cnv", variable_lengths=variable_lengths)

    return after_file


def test_healpix_cnv_generate(before_file, after_file, request):
    if before_file is None or after_file is None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    # This was done here to make it work in GitHub Actions
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("pyfhd.resources.test_data").joinpath(
            "healpix", "healpix_cnv_generate"
        )
        before_file = Path(data_dir, before_file.name)
        after_file = Path(data_dir, after_file.name)

    h5_before = load(before_file)
    expected_hpx_cnv = load(after_file)

    # Astropy strictly expects the ctype to be a string, not a numpy string
    h5_before["obs"]["astr"]["ctype"] = [
        str(x) for x in h5_before["obs"]["astr"]["ctype"]
    ]

    hpx_cnv, obs = healpix_cnv_generate(
        h5_before["obs"],
        h5_before["mask"],
        h5_before["hpx_radius"],
        h5_before["pyfhd_config"],
        Logger(1),
        nside=int(h5_before["nside"]) if "nside" in h5_before else None,
    )
    assert obs["healpix"]["nside"] == hpx_cnv["nside"]
    # restrict_healpix_inds is always True for the tests so ind_list is false
    assert obs["healpix"]["ind_list"] is None
    npt.assert_equal(obs["healpix"]["n_pix"], np.size(hpx_cnv["inds"]))
    # Unfortunately there is some differences in the FHD and AstroPy when dealing with
    # sky to pixel coordinates. The differences are small, in fact exactly one cdelt off in both
    # x and y (so off by one pixel in each axis). What I can probably do, is to ensure the indexes from
    # the expected array are in the output array, and then check the values of those found indexes. I can
    # do this because the minimum and maximum bins are beyond what FHD finds, but in theory the indexes should
    # still be in there, it's just that pyfhd finds more.
    assert hpx_cnv["nside"] == expected_hpx_cnv["nside"]
    # The indices come from a file, they should be the same
    npt.assert_array_equal(hpx_cnv["inds"], expected_hpx_cnv["inds"])
    # Hopefully once we fix the world_to_pixel issue, the following will pass
    npt.assert_array_equal(hpx_cnv["i_use"], expected_hpx_cnv["i_use"])
    npt.assert_array_equal(
        np.hstack(hpx_cnv["ija"]), np.hstack(expected_hpx_cnv["ija"])
    )
    npt.assert_allclose(
        np.hstack(hpx_cnv["sa"]), np.hstack(expected_hpx_cnv["sa"]), atol=2e-5
    )

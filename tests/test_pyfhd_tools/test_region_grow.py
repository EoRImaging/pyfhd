from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from pyfhd.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from pyfhd.pyfhd_tools.pyfhd_utils import region_grow


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "pyfhd_tools", "region_grow")


@pytest.fixture(
    scope="function",
    params=["1088285600", "1088716296", "point_zenith", "point_offzenith"],
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3", "run4"])
def run(request):
    return request.param


@pytest.fixture(scope="function", params=["", "beam_image_cube"])
def subfunc(request):
    return request.param


skip_tests = [
    ["1088285600", "", "run4"],
    ["point_zenith", "", "run3"],
    ["point_offzenith", "", "run3"],
    ["1088716296", "", "run3"],
    ["1088285600", "beam_image_cube", "run4"],
    ["point_zenith", "beam_image_cube", "run3"],
    ["point_offzenith", "beam_image_cube", "run3"],
    ["1088716296", "beam_image_cube", "run3"],
]


@pytest.fixture
def before_file(tag, run, subfunc, data_dir):
    if [tag, subfunc, run] in skip_tests:
        return None
    before_file = Path(
        data_dir,
        f"{tag}_{run}{f'_{subfunc}' if subfunc != '' else ''}_before_{data_dir.name}.h5",
    )
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)
    if subfunc == "":
        sav_dict["dimension"] = int(sav_dict["dimension"])
        sav_dict["elements"] = int(sav_dict["dimension"])
    else:
        sav_dict["b_i"] = int(sav_dict["b_i"])

    save(before_file, sav_dict, "save_dict")

    return before_file


@pytest.fixture()
def after_file(tag, run, subfunc, data_dir):
    if [tag, subfunc, run] in skip_tests:
        return None
    after_file = Path(
        data_dir,
        f"{tag}_{run}{f'_{subfunc}' if subfunc != '' else ''}_after_{data_dir.name}.h5",
    )
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    save(after_file, sav_dict["beam_i"], "beam_i")

    return after_file


@pytest.mark.github_actions
def test_simple_1D_region_grow():
    # Equivalent to doing region_grow([0,0,0,0,5,10,5,0,0,0,0], [5], threshold=[5,10])
    input = np.array([0, 0, 0, 0, 5, 10, 5, 0, 0, 0, 0])
    expected = np.array([4, 5, 6])
    output = region_grow(input, [5], low=5, high=10)

    npt.assert_array_equal(output, expected)


@pytest.mark.github_actions
def test_simple_2D_region_grow():
    # Equivalent to doing:
    # IDL> test = reform(indgen(100)*1., 10,10)
    # IDL> region_grow(test, indgen(10)+45, threshold=[30,70])
    input = np.arange(100).reshape([10, 10])
    expected = np.array(
        [
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
        ]
    )
    output = region_grow(input, np.arange(45, 55), low=30, high=70)

    npt.assert_array_equal(output, expected)


def test_FHD_region_grow(before_file, after_file):
    if before_file is None or after_file is None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )
    before = load(before_file)
    expected_beam_i = load(after_file)
    if "beam_image" in before_file.name:
        # Do the region_grow call from beam_image_cube
        beam_i = region_grow(
            before["beam_single"],
            before["b_i"],
            low=before["beam_threshold"] ** (before["square"] + 1),
            high=np.max(before["beam_single"]),
        )
    else:
        # Do the region_grow call from fhd_quickview
        beam_i = region_grow(
            before["beam_mask_test"],
            int(before["dimension"] / 2 + before["dimension"] * before["elements"] / 2),
            low=before["beam_output_threshold"],
            high=np.max(before["beam_mask_test"]),
        )

    npt.assert_array_equal(beam_i, expected_beam_i)

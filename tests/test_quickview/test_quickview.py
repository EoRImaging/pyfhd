from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from logging import Logger
import astropy
from PyFHD.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from PyFHD.io.pyfhd_quickview import quickview


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "quickview", "fhd_quickview")


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
    # If the h5 file already exists and has been created, check if it's got a
    # valid path. if it does, return it, otherwise remake it.
    if before_file.exists():
        h5_before = load(before_file)
        if Path(h5_before["pyfhd_config"]["output_dir"]).exists():
            return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    del sav_dict["cal"]
    del sav_dict["skymodel"]
    del sav_dict["jones"]
    del sav_dict["file_path_fhd"]
    del sav_dict["ring_radius"]
    sav_dict = recarray_to_dict(sav_dict)

    # Set the save everything except the fits we're testing to false
    pyfhd_config = {
        "obs_id": (
            "1088716176"
            if "point" in before_file.name
            else before_file.name.split("_")[0]
        ),
        "output_dir": Path(data_dir, f"{tag}_{run}_test_data"),
        "save_obs": False,
        "save_params": False,
        "save_visibilities": False,
        "save_cal": False,
        "save_weights": False,
        "pad_uv_image": sav_dict["pad_uv_image"],
        "image_filter": "filter_uv_uniform",
        "allow_sidelobe_model_sources": bool(
            sav_dict["extra"]["allow_sidelobe_model_sources"]
        ),
    }

    sav_dict["pyfhd_config"] = pyfhd_config
    sav_dict["model_uv"] = sav_dict["extra"]["model_uv_holo"]

    save(before_file, sav_dict, "sav_dict")

    return before_file


def test_quickview(before_file, data_dir):
    if before_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    psf = load(
        Path(env.get("PYFHD_TEST_PATH"), "beams", "decomp_beam_pointing0.h5"),
        lazy_load=True,
    )

    # Make our test directory for saving the FITS files
    h5_before["pyfhd_config"]["output_dir"] = Path(
        h5_before["pyfhd_config"]["output_dir"]
    )
    h5_before["pyfhd_config"]["output_dir"].mkdir(exist_ok=True)

    h5_before["pyfhd_config"]["image_plots"] = False

    # Astropy needs things to be a str Python type, numpy str type no longer works?
    h5_before["obs"]["astr"]["ctype"] = [
        str(x) for x in h5_before["obs"]["astr"]["ctype"]
    ]

    h5_before["obs"]["baseline_info"]["fbin_i"] = h5_before["obs"]["baseline_info"][
        "fbin_i"
    ].astype(np.int64)

    h5_before["obs"]["obsx"] = int(h5_before["obs"]["obsx"])
    h5_before["obs"]["obsy"] = int(h5_before["obs"]["obsy"])

    # Since we're not saving anything other than the fits we can ignore everything
    # else but the things required to make the FITS files. In this case we're lazy loading
    # beam file to save on memory.
    quickview(
        h5_before["obs"],
        psf,
        None,
        None,
        None,
        None,
        h5_before["image_uv_arr"],
        h5_before["weights_arr"],
        None,
        None,
        h5_before["model_uv"],
        h5_before["pyfhd_config"],
        Logger(1),
    )
    # Close the HDF5 file
    psf.close()
    # Check all the FITS Files
    fits_path = h5_before["pyfhd_config"]["output_dir"] / "fits"
    files = fits_path.glob("*.fits")
    fits_test_dir = "_".join(["fhd"] + before_file.name.split("_")[:-3])
    fits_test_dir = Path(data_dir, fits_test_dir)
    for fits in files:
        name = fits.name
        name = name.replace("beam", "Beam")
        name = name.replace("dirty", "Dirty")
        name = name.replace("model", "Model")
        name = name.replace("residual", "Residual")
        name = name.replace("uv", "UV")
        expected_fits = Path(fits_test_dir, name)
        expected_fits = astropy.io.fits.open(expected_fits)
        actual_fits = astropy.io.fits.open(fits)
        actual_header = actual_fits[0].header
        expected_header = expected_fits[0].header
        # Remove keys that are different between the two FITS files
        if expected_header.get("COMMENT") is not None:
            del expected_header["COMMENT"]
        if expected_header.get("DATE") is not None:
            del expected_header["DATE"]
        if expected_header.get("EXTEND") is not None:
            del expected_header["EXTEND"]
        if actual_header.get("EXTEND") is not None:
            del actual_header["EXTEND"]
        if actual_header.get("HISTORY") is not None:
            del actual_header["HISTORY"]
        if expected_header.get("HISTORY") is not None:
            del expected_header["HISTORY"]
        header_diff = astropy.io.fits.HeaderDiff(
            actual_header,
            expected_header,
            atol=2e-4,
            ignore_comments=["*"],
        )
        print(header_diff.report())
        # print(actual_header)
        # print(expected_header)
        assert header_diff.identical
        # expected_fits_data = expected_fits[0].data
        # actual_fits_data = actual_fits[0].data
        # expected_fits_data = np.transpose(expected_fits_data)
        # npt.assert_allclose(actual_fits_data, expected_fits_data, atol=1e-6)

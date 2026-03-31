import pytest
import numpy.testing as npt
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.source_modeling.vis_model_transfer import (
    vis_model_transfer,
)
from PyFHD.io.pyfhd_io import save, load, recarray_to_dict
from PyFHD.io.pyfhd_io import convert_sav_to_dict
from logging import Logger


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "source_modeling", "vis_model_transfer")


@pytest.fixture
def model_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "source_modeling", "models")


@pytest.fixture(scope="function", params=["point_zenith", "1088716296", "1088285600"])
def tag(request):
    return request.param


# Model imports are the same between runs, only grab run2
@pytest.fixture(scope="function", params=["run2"])
def run(request):
    return request.param


@pytest.fixture()
def before_file(tag, run, data_dir, model_dir):
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, check if it's got a
    # valid path. if it does, return it, otherwise remake it.
    if before_file.exists():
        h5_before = load(before_file)
        if Path(h5_before["pyfhd_config"]["model_file_path"]).exists():
            return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    h5_save_dict = recarray_to_dict(sav_dict)
    del h5_save_dict["model_transfer"]
    # For point_zenith let's test the uvfits we have
    h5_save_dict["pyfhd_config"] = {
        "flag_model": False if "skip_model_flagging" in h5_save_dict["extra"] else True,
        "model_file_type": "uvfits" if tag == "point_zenith" else "sav",
        "model_file_path": (
            str(Path(model_dir, f"{tag}.uvfits"))
            if tag == "point_zenith"
            else str(model_dir)
        ),
        "obs_id": tag,
        "instrument": "mwa",
        "n_pol": 2,
        "save_model": False,
    }

    del h5_save_dict["extra"]

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture()
def after_file(tag, run, data_dir):
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    vis_model_arr = np.empty(
        [
            sav_dict["vis_model_arr"].shape[0],
            sav_dict["vis_model_arr"][0].shape[1],
            sav_dict["vis_model_arr"][0].shape[0],
        ],
        dtype=np.complex128,
    )

    for pol_i in range(sav_dict["vis_model_arr"].shape[0]):
        vis_model_arr[pol_i] = sav_dict["vis_model_arr"][pol_i].T

    save(after_file, vis_model_arr, "vis_model_arr")

    return after_file


def test_model_transfer(before_file, after_file):
    h5_before = load(before_file)
    expected_vis_model_arr = load(after_file)

    vis_model_arr = vis_model_transfer(
        h5_before["pyfhd_config"], h5_before["obs"], h5_before["params"], Logger(1)
    )
    npt.assert_allclose(vis_model_arr, expected_vis_model_arr, atol=1e-8)

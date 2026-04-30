import pytest
import numpy as np
import numpy.testing as npt
from os import environ as env
from pathlib import Path
import logging
from pyfhd.data_setup.uvfits import extract_header, extract_visibilities
from pyfhd.pyfhd_tools.test_utils import get_savs


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "data_setup", "uvfits_read/")


@pytest.fixture
def uvfits_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "data_setup", "uvfits/")


def test_1061316296(data_dir, uvfits_dir):
    dummy_log = logging.getLogger("dummy")
    pyfhd_config = np.load(Path(data_dir, "config.npy"), allow_pickle=True).item()
    pyfhd_config["input_path"] = Path(uvfits_dir, "1061316296")
    pyfhd_header, fits_data, _, _ = extract_header(pyfhd_config, dummy_log)
    vis_arr, vis_weights = extract_visibilities(
        pyfhd_header, fits_data, pyfhd_config, dummy_log
    )

    output = get_savs(data_dir, "output.sav")

    for pol_i in range(pyfhd_config["n_pol"]):
        to_shape = (vis_arr.shape[1], vis_arr.shape[-1])
        expected_vis_arr = output["vis_arr"][pol_i].transpose()
        expected_vis_weights = output["vis_weights"][pol_i].transpose()
        npt.assert_allclose(vis_arr[pol_i, :, :], expected_vis_arr)
        npt.assert_allclose(vis_weights[pol_i, :, :], expected_vis_weights)

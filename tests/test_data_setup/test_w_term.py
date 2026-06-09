import pytest
import numpy as np
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from logging import Logger
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.pyfhd_tools.pyfhd_utils import simple_deproject_w_term
from pyfhd.pyfhd_tools.test_utils import get_savs


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "data_setup", "simple_deproject_w_term/")


def test_w_term_1(data_dir):
    dummy_log = Logger(1)
    inputs = get_savs(data_dir, "input_1.sav")
    inputs = recarray_to_dict(inputs)
    vis_arr = np.moveaxis(inputs["vis_arr"], 1, -1)
    vis_arr = simple_deproject_w_term(
        inputs["obs"], inputs["params"], vis_arr, inputs["direction"], dummy_log
    )
    expected_output = get_savs(data_dir, "output_1.sav")
    for pol_i in range(inputs["obs"]["n_pol"]):
        npt.assert_allclose(vis_arr[pol_i, :, :], expected_output["vis_arr"][pol_i].T)


def test_w_term_2(data_dir):
    dummy_log = Logger(1)
    inputs = get_savs(data_dir, "input_2.sav")
    inputs = recarray_to_dict(inputs)
    vis_arr = np.moveaxis(inputs["vis_arr"], 1, -1)
    vis_arr = simple_deproject_w_term(
        inputs["obs"], inputs["params"], vis_arr, inputs["direction"], dummy_log
    )
    expected_output = get_savs(data_dir, "output_2.sav")
    for pol_i in range(inputs["obs"]["n_pol"]):
        npt.assert_allclose(vis_arr[pol_i, :, :], expected_output["vis_arr"][pol_i].T)

from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from pyfhd.pyfhd_tools.test_utils import (
    get_data,
    get_data_items,
    sav_file_vis_arr_swap_axes,
)
from pyfhd.gridding.gridding_utils import visibility_count
from pyfhd.io.pyfhd_io import save, load
from logging import Logger
from numpy.testing import assert_allclose


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "gridding", "visibility_count")


@pytest.fixture(scope="function", params=[1, 2, 3])
def number(request):
    return request.param


def get_file(data_dir, file_name):
    if Path(data_dir, file_name).exists():
        item = get_data_items(data_dir, file_name)
        return item
    else:
        return None


@pytest.fixture
def vis_count_before(data_dir, number):
    vis_count_before = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if vis_count_before.exists():
        return vis_count_before

    # First put values from psf into pyfhd_config
    psf = get_data(
        data_dir,
        f"input_psf_{number}.npy",
    )
    psf[0]["beam_ptr"] = psf[0]["beam_ptr"][0, :, :].T
    psf = recarray_to_dict(psf)
    # Take the required parameters
    obs, params, vis_weights = get_data_items(
        data_dir,
        f"input_obs_{number}.npy",
        f"input_params_{number}.npy",
        f"input_vis_weight_ptr_{number}.npy",
    )

    # One of the tests accidentally gave the full vis_weights,
    # grab the first polarization as that is the one that was used
    # for test 2.
    if vis_weights.shape[0] == 2:
        vis_weights = vis_weights[0]

    # Create the save dict
    h5_save_dict = {}
    h5_save_dict["obs"] = recarray_to_dict(obs)
    h5_save_dict["psf"] = psf
    h5_save_dict["params"] = recarray_to_dict(params)
    h5_save_dict["vis_weights"] = vis_weights.transpose()
    h5_save_dict["fi_use"] = get_file(data_dir, f"input_fi_use_{number}.npy")
    h5_save_dict["bi_use"] = get_file(data_dir, f"input_bi_use_arr_{number}.npy")
    h5_save_dict["mask_mirror_indices"] = (
        True if get_file(data_dir, f"input_mask_mirror_indices_{number}.npy") else False
    )
    h5_save_dict["no_conjugate"] = (
        True if get_file(data_dir, f"input_no_conjugate_{number}.npy") else False
    )
    h5_save_dict["fill_model_visibilities"] = (
        True if get_file(data_dir, f"input_fill_model_vis_{number}.npy") else False
    )

    # Save it
    save(vis_count_before, h5_save_dict, "before_file")

    return vis_count_before


@pytest.fixture
def vis_count_after(data_dir, number):
    vis_count_after = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if vis_count_after.exists():
        return vis_count_after
    uniform_filter = get_data_items(
        data_dir,
        f"output_uniform_filter_{number}.npy",
    )

    h5_save_dict = {"uniform_filter": uniform_filter}

    save(vis_count_after, h5_save_dict, "after_file")

    return vis_count_after


def test_vis_count(vis_count_before: Path, vis_count_after: Path):
    h5_before = load(vis_count_before)
    expected_uniform_filter = load(vis_count_after)

    uniform_filter = visibility_count(
        h5_before["obs"],
        h5_before["psf"],
        h5_before["params"],
        h5_before["vis_weights"],
        Logger(1),
        fi_use=h5_before["fi_use"],
        bi_use=h5_before["bi_use"],
        mask_mirror_indices=h5_before["mask_mirror_indices"],
        no_conjugate=h5_before["no_conjugate"],
        fill_model_visibilities=h5_before["fill_model_visibilities"],
    )
    # Due to precision errors from baseline_grid_locations the check for
    # visibility_count has to be almost the same as xmin and ymin are used
    # directly on uniform filter and can be off by 1. Should also note that
    # the values are only different for a maximum of 0.0207% of the values when
    # considering the differences directly, when rtol = 1 the differences are
    # account for 9.54e-05% of the dataset, which is why the atol is 1.5
    # specifically for test 3. The relative and absolute tolerance can be lower for
    # tests 1 and 2
    assert_allclose(uniform_filter, expected_uniform_filter, rtol=1, atol=1.51)

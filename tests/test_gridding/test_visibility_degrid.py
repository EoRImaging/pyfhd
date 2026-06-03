from logging import Logger
from os import environ as env
from pathlib import Path

import pytest
import numpy.testing as npt

from pyfhd.gridding.visibility_degrid import visibility_degrid
from pyfhd.io.pyfhd_io import load, recarray_to_dict, save
from pyfhd.pyfhd_tools.test_utils import get_savs, sav_file_rearrange_psf


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "gridding/visibility_degrid/")


@pytest.fixture(
    scope="function",
    params=[1, 2],
    #     pytest.param(1, marks=pytest.mark.github_actions),
    #     pytest.param(2, marks=pytest.mark.github_actions),
    #     pytest.param(3, marks=pytest.mark.github_actions),
    # ],
)
def number(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def before_degridding(data_dir: Path, number: int, request: pytest.FixtureRequest):
    before_gridding = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if before_gridding.exists():
        return before_gridding

    if number == 3:
        # This is not currently run. It doesn't work, seems like the data isn't
        # properly set up for beam_per_baseline. But that option is largely abandoned,
        # so not worrying about testing it for now.
        # Note that if beam_per_baseline is False, set 3 is identical to set 1.
        beam_per_baseline = True
    else:
        beam_per_baseline = False

    h5_save_dict = get_savs(data_dir, f"input_{number}.sav")

    # fix the psf to be properly arranged
    h5_save_dict["psf"] = sav_file_rearrange_psf(h5_save_dict["psf"])

    h5_save_dict = recarray_to_dict(h5_save_dict)
    h5_save_dict["obs"]["n_baselines"] = h5_save_dict["obs"]["nbaselines"]
    # Transpose the model if it exists
    h5_save_dict["pyfhd_config"] = {
        "interpolate_kernel": h5_save_dict["psf"]["interpolate_kernel"],
        "psf_dim": h5_save_dict["psf"]["dim"],
        "psf_resolution": h5_save_dict["psf"]["resolution"],
        "beam_mask_threshold": h5_save_dict["psf"]["beam_mask_threshold"],
        "beam_clip_floor": h5_save_dict["extra"]["beam_clip_floor"],
        "beam_per_baseline": beam_per_baseline,
        # need this to be defined (not actually used)
        "image_filter": "filter_uv_uniform",
        # "grid_spectral": (
        #     True
        #     if ("grid_spectral" in h5_save_dict and h5_save_dict["grid_spectral"])
        #     else False
        # ),
    }

    h5_save_dict["vis_weight_ptr"] = h5_save_dict["vis_weight_ptr"].T
    h5_save_dict["image_uv"] = h5_save_dict["image_uv"].T

    h5_save_dict["vis_input"] = None
    h5_save_dict["beam_per_baseline"] = beam_per_baseline

    if h5_save_dict["conserve_memory"] > 1e6:
        h5_save_dict["memory_threshold"] = h5_save_dict["conserve_memory"]
        h5_save_dict["conserve_memory"] = True
    elif h5_save_dict["conserve_memory"] > 0:
        h5_save_dict["memory_threshold"] = 1e8
        h5_save_dict["conserve_memory"] = True
    else:
        h5_save_dict["conserve_memory"] = False

    save(before_gridding, h5_save_dict, "before_file")

    return before_gridding


@pytest.fixture
def after_degridding(data_dir: Path, number: int, request: pytest.FixtureRequest):
    after_gridding = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if after_gridding.exists():
        return after_gridding

    outputs = get_savs(data_dir, f"output_{number}_new.sav")
    outputs = recarray_to_dict(outputs)

    h5_save_dict = {
        "vis_return": outputs["vis_return"].T,
    }

    save(after_gridding, h5_save_dict, "after_file")

    return after_gridding


def test_visibility_degrid(
    before_degridding: Path, after_degridding: Path, request: pytest.FixtureRequest
):
    h5_before = load(before_degridding)
    vis_expected = load(after_degridding)

    vis_return = visibility_degrid(
        image_uv=h5_before["image_uv"],
        vis_weights=h5_before["vis_weight_ptr"],
        obs=h5_before["obs"],
        psf=h5_before["psf"],
        params=h5_before["params"],
        pyfhd_config=h5_before["pyfhd_config"],
        logger=Logger(1),
        polarization=h5_before["polarization"],
        fill_model_visibilities=h5_before["fill_model_visibilities"],
        vis_input=h5_before["vis_input"],
        spectral_model_uv_arr=h5_before["spectral_model_uv_arr"],
        beam_per_baseline=h5_before["beam_per_baseline"],
        conserve_memory=h5_before["conserve_memory"],
        memory_threshold=h5_before["memory_threshold"],
    )

    npt.assert_allclose(vis_return, vis_expected, atol=1e-15)

import pytest
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from pyfhd.gridding.visibility_degrid import visibility_degrid
from pyfhd.pyfhd_tools.test_utils import get_savs


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "visibility_degrid/")


@pytest.fixture
def full_data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "full_size_visibility_degrid/")


def _test_visibility_degrid_one(data_dir):
    inputs = get_savs(data_dir, "input_1.sav")
    image_uv = inputs["image_uv"]
    vis_weights = inputs["vis_weight_ptr"]
    obs = inputs["obs"]
    psf = inputs["psf"]
    params = inputs["params"]
    polarization = inputs["polarization"]
    fill_model_visibilities = inputs["fill_model_visibilities"]
    vis_input = None
    spectral_model_uv_arr = inputs["spectral_model_uv_arr"]
    beam_per_baseline = False
    uv_grid_phase_only = True
    conserve_memory = inputs["conserve_memory"]
    memory_threshold = 1e8

    vis_return = visibility_degrid(
        image_uv,
        vis_weights,
        obs,
        psf,
        params,
        polarization=polarization,
        fill_model_visibilities=fill_model_visibilities,
        vis_input=vis_input,
        spectral_model_uv_arr=spectral_model_uv_arr,
        beam_per_baseline=beam_per_baseline,
        uv_grid_phase_only=uv_grid_phase_only,
        conserve_memory=conserve_memory,
        memory_threshold=memory_threshold,
    )

    outputs = get_savs(data_dir, "output_1.sav")

    npt.assert_allclose(vis_return, outputs["vis_return"], atol=1e-3)


def _test_visibility_degrid_two(data_dir):

    inputs = get_savs(data_dir, "input_2.sav")
    image_uv = inputs["image_uv"]
    vis_weights = inputs["vis_weight_ptr"]
    obs = inputs["obs"]
    psf = inputs["psf"]
    params = inputs["params"]
    polarization = inputs["polarization"]
    fill_model_visibilities = inputs["fill_model_visibilities"]
    vis_input = None
    spectral_model_uv_arr = inputs["spectral_model_uv_arr"]
    beam_per_baseline = False
    uv_grid_phase_only = True
    conserve_memory = inputs["conserve_memory"]
    memory_threshold = 1e8

    vis_return = visibility_degrid(
        image_uv,
        vis_weights,
        obs,
        psf,
        params,
        polarization=polarization,
        fill_model_visibilities=fill_model_visibilities,
        vis_input=vis_input,
        spectral_model_uv_arr=spectral_model_uv_arr,
        beam_per_baseline=beam_per_baseline,
        uv_grid_phase_only=uv_grid_phase_only,
        conserve_memory=conserve_memory,
        memory_threshold=memory_threshold,
    )

    outputs = get_savs(data_dir, "output_2.sav")

    npt.assert_allclose(vis_return, outputs["vis_return"], atol=9e-6)


def _test_visibility_degrid_three(data_dir):

    inputs = get_savs(data_dir, "input_3.sav")
    image_uv = inputs["image_uv"]
    vis_weights = inputs["vis_weight_ptr"]
    obs = inputs["obs"]
    psf = inputs["psf"]
    params = inputs["params"]
    polarization = inputs["polarization"]
    fill_model_visibilities = inputs["fill_model_visibilities"]
    vis_input = None
    spectral_model_uv_arr = inputs["spectral_model_uv_arr"]
    beam_per_baseline = False
    uv_grid_phase_only = True
    conserve_memory = inputs["conserve_memory"]
    memory_threshold = 1e8

    vis_return = visibility_degrid(
        image_uv,
        vis_weights,
        obs,
        psf,
        params,
        polarization=polarization,
        fill_model_visibilities=fill_model_visibilities,
        vis_input=vis_input,
        spectral_model_uv_arr=spectral_model_uv_arr,
        beam_per_baseline=beam_per_baseline,
        uv_grid_phase_only=uv_grid_phase_only,
        conserve_memory=conserve_memory,
        memory_threshold=memory_threshold,
    )

    outputs = get_savs(data_dir, "output_3.sav")

    npt.assert_allclose(vis_return, outputs["vis_return"], atol=9e-6)

from logging import Logger
from os import environ as env
from pathlib import Path

import pytest
import numpy.testing as npt

from pyfhd.gridding.visibility_degrid import visibility_degrid
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.pyfhd_tools.test_utils import get_savs


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "gridding/visibility_degrid/")


def test_visibility_degrid_one(data_dir):
    inputs = get_savs(data_dir, "input_1.sav")
    inputs = recarray_to_dict(inputs)
    image_uv = inputs["image_uv"]
    vis_weights = inputs["vis_weight_ptr"].transpose()
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
    if conserve_memory > 1e6:
        memory_threshold = conserve_memory
        conserve_memory = True
    elif conserve_memory > 0:
        memory_threshold = 1e8
        conserve_memory = True
    else:
        conserve_memory = False

    vis_return = visibility_degrid(
        image_uv,
        vis_weights,
        obs,
        psf,
        params,
        Logger(1),
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
    outputs = recarray_to_dict(outputs)

    npt.assert_allclose(vis_return, outputs["vis_return"].T, atol=9e-6)


def test_visibility_degrid_two(data_dir):

    inputs = get_savs(data_dir, "input_2.sav")
    inputs = recarray_to_dict(inputs)
    image_uv = inputs["image_uv"]
    vis_weights = inputs["vis_weight_ptr"].transpose()
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
    if conserve_memory > 1e6:
        memory_threshold = conserve_memory
        conserve_memory = True
    elif conserve_memory > 0:
        memory_threshold = 1e8
        conserve_memory = True
    else:
        conserve_memory = False

    vis_return = visibility_degrid(
        image_uv,
        vis_weights,
        obs,
        psf,
        params,
        Logger(1),
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
    outputs = recarray_to_dict(outputs)

    npt.assert_allclose(vis_return, outputs["vis_return"].T, atol=9e-6)


def test_visibility_degrid_three(data_dir):

    inputs = get_savs(data_dir, "input_3.sav")
    inputs = recarray_to_dict(inputs)
    image_uv = inputs["image_uv"]
    vis_weights = inputs["vis_weight_ptr"].transpose()
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
    if conserve_memory > 1e6:
        memory_threshold = conserve_memory
        conserve_memory = True
    elif conserve_memory > 0:
        memory_threshold = 1e8
        conserve_memory = True
    else:
        conserve_memory = False

    vis_return = visibility_degrid(
        image_uv,
        vis_weights,
        obs,
        psf,
        params,
        Logger(1),
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
    outputs = recarray_to_dict(outputs)

    npt.assert_allclose(vis_return, outputs["vis_return"].T, atol=9e-6)

import copy

import numpy as np
import pytest
from pyradiosky import SkyModel
from pyuvdata.datasets import fetch_data as uvdata_fetch

from pyfhd.beam_setup.beam import create_psf
from pyfhd.data.datasets import fetch_data
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.pyfhd_tools.test_utils import get_savs, sav_file_rearrange_psf
from pyfhd.source_modeling.source_utils import source_dft


@pytest.fixture(scope="session")
def zenith_obs_2013_main():

    obs_file = fetch_data("2013_zenith_obs_2times")

    obs_sav_dict = get_savs(obs_file, "")
    obs_sav_dict = recarray_to_dict(obs_sav_dict)
    obs = obs_sav_dict["obs"]
    obs["n_baselines"] = obs["nbaselines"]
    obs["delays"] = obs["delays"].astype("int").repeat(2).reshape((2, 16))
    obs["dimension"] = int(obs["dimension"])
    obs["elements"] = int(obs["elements"])

    yield obs


@pytest.fixture(scope="function")
def zenith_obs_2013(zenith_obs_2013_main):
    obs_in = zenith_obs_2013_main

    obs = copy.deepcopy(obs_in)

    yield obs


@pytest.fixture(scope="session")
def zenith_params_2013_main():
    params_path = fetch_data("2013_zenith_params_2times")

    params_sav_dict = get_savs(params_path, "")
    params_sav_dict = recarray_to_dict(params_sav_dict)
    params = params_sav_dict["params"]

    yield params


@pytest.fixture(scope="function")
def zenith_params_2013(zenith_params_2013_main):
    params_in = zenith_params_2013_main

    params = copy.deepcopy(params_in)

    yield params


@pytest.fixture(scope="session")
def zenith_psf_2013_cut():
    psf_cut_file = fetch_data("2013_zenith_psf_small")

    psf_sav_dict = get_savs(psf_cut_file, "")
    # fix the psf to be properly arranged
    psf = sav_file_rearrange_psf(psf_sav_dict["psf"])

    yield psf


@pytest.fixture(scope="session")
def mwa_aee_beam_zenith_2013(zenith_obs_2013_main, tmp_path_factory):
    obs = copy.deepcopy(zenith_obs_2013_main)

    # check to make sure we have the right version of pyuvdata installed
    from pyuvdata import UVBeam

    assert hasattr(UVBeam, "decompose_feed_aligned_terms")

    mwa_aee_jfile = uvdata_fetch("mwa_jmatrix")
    mwa_aee_zfile = uvdata_fetch("mwa_zmatrix")

    # set obs freq_array to match uvbeam freq_array
    freqs = np.array([1.6512e08, 1.8048e08])
    n_freq = freqs.size
    obs["n_freq"] = n_freq
    # the input nf_vis has a pol axis, which we don't want.
    obs["nf_vis"] = obs["nf_vis"][0:n_freq, 0]
    obs["freq_center"] = np.mean(freqs)
    obs["baseline_info"]["freq"] = freqs
    obs["baseline_info"]["freq_use"] = np.ones((n_freq,), dtype=int)
    obs["baseline_info"]["fbin_i"] = np.arange(n_freq, dtype=int)

    pyfhd_config = {
        "instrument": "mwa",
        "psf_dim": 14,
        "psf_resolution": 10,
        "beam_mask_threshold": 1e2,
        "uvbeam_file_path": mwa_aee_jfile,
        "uvbeam_zfile_path": mwa_aee_zfile,
        "interpolate_kernel": False,
        "beam_offset_time": None,
        "analytic_beam_yaml": None,
        "uvbeam_freq_buffer": 2e6,
        "beam_clip_floor": False,
        "uvbeam_mwa_include_cross_feed_coupling": False,
        "output_dir": tmp_path_factory.mktemp("zenith_2013"),
        "obs_id": 1061316296,
        "cal_stop": False,
        "conserve_memory": True,
        "memory_threshold": 1e10,
    }

    psf, antenna = create_psf(obs, pyfhd_config)

    yield antenna, psf, obs, pyfhd_config


@pytest.fixture(scope="session")
def model_uv_zenith_2013():
    idl_cal_sources_file = fetch_data("2013_zenith_gleam_sources")

    # This particular file doesn't have XX & YY fluxes (they're all zero)
    # So just use I flux/2 because we're just checking our dft code.
    sky = SkyModel.from_file(
        idl_cal_sources_file, extra_columns={"x": "image_x", "y": "image_y"}
    )

    # take this out once pyradiosky fix is in:
    if sky.extended_model_group is not None:
        extended_comps = np.nonzero(sky.extended_model_group != "")[0]
        if extended_comps.size == 0:
            sky.extended_model_group = None

    sky.at_frequencies(np.atleast_1d(sky.reference_frequency[0]))

    dim_use = 2048

    x_vec = sky.extra_columns["image_x"].astype(np.float64)
    y_vec = sky.extra_columns["image_y"].astype(np.float64)

    # use full plane
    uv_mask = np.full((dim_use, dim_use), True)

    uv_i_inds = np.nonzero(uv_mask)

    xinds, yinds = np.meshgrid(np.arange(dim_use), np.arange(dim_use))

    xinds = xinds[uv_i_inds]
    yinds = yinds[uv_i_inds]
    xinds = xinds.flatten()
    yinds = yinds.flatten()

    xvals = xinds - dim_use / 2.0
    yvals = yinds - dim_use / 2.0

    flux_arr = np.zeros((1, sky.Ncomponents), dtype=np.float64)
    flux_arr[0] = sky.stokes[0]

    model_uv_vals = source_dft(
        x_loc=x_vec,
        y_loc=y_vec,
        xvals=xvals,
        yvals=yvals,
        dimension=dim_use,
        elements=dim_use,
        flux=flux_arr,
        memory_threshold=5e9,  # chosen to exercise parts of the code
        conserve_memory=True,
    )
    model_uv_full = np.zeros((dim_use, dim_use), dtype=np.complex128)
    model_uv_full[xinds, yinds] = model_uv_vals

    yield model_uv_full

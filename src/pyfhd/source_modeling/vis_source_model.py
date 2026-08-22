from pathlib import Path
import copy
import logging
import time

import numpy as np
from pyradiosky import SkyModel

from .source_utils import source_dft_model, vis_delay_filter
from ..gridding.visibility_degrid import visibility_degrid
from ..pyfhd_tools.types import BoolArray, ComplexArray, FloatArray
from ..pyfhd_tools.pyfhd_utils import _print_time_diff
from ..io.pyfhd_io import save

logger = logging.getLogger(__name__)


def vis_source_model(
    *,
    pyfhd_config: dict,
    obs: dict,
    psf: dict,
    params: dict,
    antenna: dict,
    skymodel: SkyModel,
    vis_weights: FloatArray | None,
    uv_mask: BoolArray | None = None,
    model_delay_filter: bool = True,
    fill_model_visibilities: bool = False,
    vis_model: BoolArray | None = None,
) -> ComplexArray:
    """
    Simulate model visibilities via degridding.

    Parameters
    ----------
    pyfhd_config : dict
        PyFHD's configuration dictionary containing all the options for a PyFHD
        run.
    obs : dict
        The observation metadata dictionary.
    psf: dict
        The gridding kernel dictionary.
    params : dict
        Visibility metadata dictionary.
    antenna: dict
        The antenna/beam dictionary.
    skymodel: pyradiosky.SkyModel
        Skymodel object containing the sources to use.
    vis_weights : ndarray of float
        Weights (flags) of the visibilities. Can be None if fill_model_visibilities
        is True.
    uv_mask : ndarray of bool, optional
        Boolean mask of what parts of the uv plane to use. Defaults to using
        half the plane (with a margin for uv beam spillover).
    model_delay_filter : bool
        Option to apply a delay filter to the model visibilities. When this is
        done, the bandwidth of the simulated visibilities is doubled and then
        reduced back to the input frequency array after filtering.
    fill_model_visibilities : bool, optional
        Create all model visibilities disregarding flags, by default False
    vis_model : ndarray of complex | None, optional
        Extra model visibilities to add to the degridded products, by default None.

    Returns
    -------
    vis_model_arr : ndarray of complex
        Simulated model for the visibilities

    """
    n_pol = obs["n_pol"]
    dimension = obs["dimension"]
    elements = obs["elements"]
    n_spectral = obs["degrid_spectral_terms"]
    if n_spectral != 0:
        raise NotImplementedError("degridding spectral terms is not yet implemented.")

    if uv_mask is not None:
        uv_mask_use = uv_mask
    else:
        # mask half the uv plane by default
        uv_mask_use = np.full((dimension, elements), True)
        uv_mask_use[:, elements // 2 + psf["dim"] :] = False

    freq_bin_i = obs["baseline_info"]["fbin_i"]
    frequency_array = obs["baseline_info"]["freq"]
    nfreq_bin = np.max(freq_bin_i) + 1
    nbaselines = obs["n_baselines"]
    n_samples = obs["n_time"]
    n_freq = obs["n_freq"]

    if model_delay_filter:
        obs_use = copy.deepcopy(obs)
        psf_use = copy.deepcopy(psf)
        freq_bin_i = np.concatenate(
            (
                np.zeros(int(np.ceil(n_freq / 2)), dtype=int) + freq_bin_i[0],
                freq_bin_i,
                np.zeros(int(np.ceil(n_freq / 2)), dtype=int) + freq_bin_i[-1],
            )
        )
        freq_res = obs["freq_res"]
        low_freq = frequency_array[0] - freq_res * n_freq / 2
        frequency_array = np.arange(n_freq * 2) * freq_res + low_freq

        nfreq_bin = np.max(freq_bin_i) + 1
        n_freq = n_freq * 2
        freq_use = np.ones((n_freq), dtype=int)

        obs_use["n_freq"] = n_freq
        obs_use["baseline_info"]["freq_use"] = freq_use
        obs_use["baseline_info"]["freq"] = frequency_array
        obs_use["baseline_info"]["fbin_i"] = freq_bin_i

        psf_use["n_freq"] = nfreq_bin
        psf_use["freq_use"] = freq_use
        psf_use["freq"] = frequency_array
        psf_use["fbin_i"] = freq_bin_i
    else:
        obs_use = obs
        psf_use = psf

    vis_dimension = nbaselines * n_samples
    logger.info("Begin source DFT")
    dft_start = time.time()
    model_uv_arr = source_dft_model(
        skymodel=skymodel,
        obs=obs_use,
        antenna=antenna,
        # sigma_threshold=2.,
        uv_mask=uv_mask_use,
        conserve_memory=pyfhd_config["conserve_memory"],
        memory_threshold=pyfhd_config["memory_threshold"],
    )
    dft_end = time.time()
    _print_time_diff(dft_start, dft_end, "source DFT")

    # Option to save model uv plane as part of a calibration-only loop.
    # put this here rather than in calibrate because the model_uv plane is not
    # used in calibrate.
    if pyfhd_config["cal_stop"]:
        model_uv_path = Path(
            pyfhd_config["output_dir"],
            "calibration",
            f"{pyfhd_config['obs_id']}_model_uv_arr.h5",
        )
        logger.info(f"Saving the models uv plane to {model_uv_path}")
        save(model_uv_path, model_uv_arr, "model_uv")

    vis_arr = np.zeros((n_pol, n_freq, vis_dimension), dtype=np.cdouble)

    logger.info("Begin Degridding")
    degrid_start = time.time()
    for pol_i in range(n_pol):
        if vis_model is not None:
            vis_input = vis_model[pol_i]
        else:
            vis_input = None
        if vis_weights is not None:
            vis_weights_use = vis_weights[pol_i]
        else:
            vis_weights_use = None

        vis_arr[pol_i] = visibility_degrid(
            pyfhd_config=pyfhd_config,
            image_uv=model_uv_arr[pol_i],
            vis_weights=vis_weights_use,
            obs=obs_use,
            psf=psf_use,
            params=params,
            polarization=pol_i,
            fill_model_visibilities=fill_model_visibilities,
            vis_input=vis_input,
        )
    degrid_end = time.time()
    _print_time_diff(degrid_start, degrid_end, "Degridding")

    if model_delay_filter:
        logger.info("Applying a horizon delay filter")
        vis_arr = vis_delay_filter(vis_arr, params=params, obs=obs_use)

    return vis_arr

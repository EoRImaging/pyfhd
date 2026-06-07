import numpy as np
from pyuvdata.datasets import fetch_data as uvdata_fetch

from pyfhd.beam_setup.antenna import init_beam
from pyfhd.beam_setup.beam_utils import beam_power
from pyfhd.data.datasets import fetch_data
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.pyfhd_tools.test_utils import get_savs


def test_beam_power():
    mwa_aee_jfile = uvdata_fetch("mwa_jmatrix")
    mwa_aee_zfile = uvdata_fetch("mwa_zmatrix")

    obs_file = fetch_data("2013_zenith_obs")

    psf_superres_fhd_file = fetch_data("2013_zenith_psf_superres")

    obs_sav_dict = get_savs(obs_file, "")
    obs_sav_dict = recarray_to_dict(obs_sav_dict)
    obs = obs_sav_dict["obs"]
    obs["n_baselines"] = obs["nbaselines"]
    obs["delays"] = obs["delays"].astype("int").repeat(2).reshape((2, 16))

    psf_superres_fhd_dict = get_savs(psf_superres_fhd_file, "")
    psf_superres_fhd_dict = recarray_to_dict(psf_superres_fhd_dict)
    psf_superres_fhd = psf_superres_fhd_dict["psf_base_superres"].T

    # set obs freq_array to match uvbeam freq_array
    freqs = np.array([1.6512e08, 1.8048e08])
    n_freq = freqs.size
    obs["n_freq"] = n_freq
    obs["nf_vis"] = obs["nf_vis"][0:n_freq, :]
    obs["freq_center"] = np.mean(freqs)
    obs["baseline_info"]["freq"] = freqs
    obs["baseline_info"]["freq_use"] = np.ones((n_freq,), dtype=int)
    obs["baseline_info"]["fbin_i"] = np.arange(n_freq, dtype=int)

    pyfhd_config = {
        "instrument": "mwa",
        "beam_model_version": "aee",
        "psf_dim": 14,
        "psf_resolution": 10,
        "beam_mask_threshold": 1e2,
        "uvbeam_file_path": mwa_aee_jfile,
        "uvbeam_zfile_path": mwa_aee_zfile,
        "beam_offset_time": None,
        "analytic_beam_yaml": None,
        "uvbeam_freq_buffer": 2e6,
        "beam_clip_floor": False,
        "uvbeam_mwa_include_cross_feed_coupling": False,
    }

    antenna, psf = init_beam(obs, pyfhd_config, None)

    zen_int_x = (obs["zenx"] - obs["obsx"]) / psf["scale"] + psf["image_dim"] / 2
    zen_int_y = (obs["zeny"] - obs["obsy"]) / psf["scale"] + psf["image_dim"] / 2

    res_super = 1 / (psf["resolution"] / psf["intermediate_res"])

    xvals_uv_superres, yvals_uv_superres = np.meshgrid(
        np.arange(psf["superres_dim"]), np.arange(psf["superres_dim"])
    )
    xvals_uv_superres = (
        xvals_uv_superres * res_super
        - np.floor(psf["dim"] / 2) * psf["intermediate_res"]
        + np.floor(psf["image_dim"] / 2)
    )
    yvals_uv_superres = (
        yvals_uv_superres * res_super
        - np.floor(psf["dim"] / 2) * psf["intermediate_res"]
        + np.floor(psf["image_dim"] / 2)
    )

    for ant_pol1 in np.arange(obs["n_pol"]):
        for ant_pol2 in np.arange(obs["n_pol"]):
            psf_base_superres = beam_power(
                antenna=antenna,
                ant_pol_1=ant_pol1,
                ant_pol_2=ant_pol2,
                freq_i=1,
                psf=psf,
                zen_int_x=zen_int_x,
                zen_int_y=zen_int_y,
                xvals_uv_superres=xvals_uv_superres,
                yvals_uv_superres=yvals_uv_superres,
                pyfhd_config=pyfhd_config,
            )

            # tolerance is not super tight because the codes used to produce
            # these are different. Among other things, the pyfhd code implements
            # the updated decomposition while the FHD code still uses the old one.
            # The frequencies are also slightly different.
            np.testing.assert_allclose(
                psf_base_superres,
                psf_superres_fhd[ant_pol1, ant_pol2],
                rtol=0,
                atol=6e-4,
            )

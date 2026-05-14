import numpy as np
from numpy.typing import NDArray
from logging import Logger
import h5py
from pathlib import Path
from pyfhd.io.pyfhd_io import save
from pyfhd.data_setup.obs import update_obs
from pyfhd.healpix.healpix_utils import (
    healpix_cnv_generate,
    healpix_cnv_apply,
    beam_image_cube,
    vis_model_freq_split,
)
from pyfhd.flagging.flagging import vis_flag_tiles
from pyfhd.pyfhd_tools.pyfhd_utils import (
    vis_weights_update,
    split_vis_weights,
    vis_noise_calc,
)


def healpix_snapshot_cube_generate(
    obs: dict,
    psf: dict | h5py.File,
    cal: dict,
    params: dict,
    vis_arr: NDArray[np.complex128],
    vis_model_arr: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    pyfhd_config: dict,
    logger: Logger,
) -> None:
    """
    Generate and save HEALPix images per polarization and per frequency channel for
    each of the gridded visibility products, creating calibrated data, model
    data, sampling map, and variance map cubes. General settings split up the
    outputs into even and odd interleaved time samples. Interpolation is
    performed from the orthoslant projection to the HEALPix grid.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary.
    psf : dict | h5py.File
        Beam dictionary
    cal : dict
        Calibration dictionary
    params : dict
        Visibility metadata dictionary
    vis_arr : NDArray[np.complex128]
        The visibility array
    vis_model_arr : NDArray[np.complex128]
        The model visibility array
    vis_weights : NDArray[np.float64]
        The visibility weights
    pyfhd_config : dict
        pyfhd configuration settings
    logger : Logger
        pyfhd's Logger
    """

    if pyfhd_config["split_ps_export"]:
        cube_name = ["hpx_even", "hpx_odd"]
    else:
        cube_name = ["healpix_cube"]

    if pyfhd_config["ps_kbinsize"] is not None:
        kbinsize = pyfhd_config["ps_kbinsize"]
    elif pyfhd_config["ps_fov"] is not None:
        # Does this get used in conjunction with the fov_use down
        # below, if so, we're converting from Radians to Degrees twice?
        kbinsize = (180 / np.pi) / pyfhd_config["ps_fov"]
    else:
        kbinsize = obs["kpix"]

    fov_use = (180 / np.pi) / kbinsize

    if pyfhd_config["ps_kspan"] is not None:
        dimension_use = pyfhd_config["ps_kspan"] / kbinsize
    elif pyfhd_config["ps_dimension"] is not None:
        dimension_use = pyfhd_config["ps_dimension"]
    elif pyfhd_config["ps_degpix"]:
        dimension_use = fov_use / pyfhd_config["ps_degpix"]
    else:
        dimension_use = fov_use / obs["degpix"]

    dimension_use = int(dimension_use)

    if pyfhd_config["ps_nfreq_avg"] is None:
        fbin_i = psf["fbin_i"]
        if isinstance(psf, h5py.File):
            fbin_i = fbin_i[:]
        ps_nfreq_avg = np.round(obs["n_freq"] / np.max(fbin_i + 1))
    else:
        ps_nfreq_avg = pyfhd_config["ps_nfreq_avg"]

    degpix_use = fov_use / dimension_use
    pix_sky = (4 * np.pi * (180 / np.pi) ** 2) / degpix_use**2
    # Below should = 1024 for 0.1119 degrees/pixel
    nside = 2 ** (np.ceil(np.log(np.sqrt(pix_sky / 12)) / np.log(2)))
    n_freq_use = int(np.floor(obs["n_freq"] / pyfhd_config["n_avg"]))

    obs_out = update_obs(
        obs, dimension_use, kbinsize, beam_nfreq_avg=ps_nfreq_avg, fov=fov_use
    )
    # To have a psf that has reacted to the new beam_nfreq_avg you have set that isn't
    # the default, tell pyfhd to re-create the psf here once beam_setup has been translated

    beam_arr, beam_mask = beam_image_cube(
        obs_out,
        psf,
        logger,
        square=True,
        beam_threshold=pyfhd_config["ps_beam_threshold"],
        n_freq_bin=n_freq_use,
    )

    hpx_radius = fov_use / np.sqrt(2)

    hpx_cnv, obs_out = healpix_cnv_generate(
        obs_out, beam_mask, hpx_radius, pyfhd_config, logger, nside=nside
    )
    hpx_inds = hpx_cnv["inds"]

    if len(pyfhd_config["ps_tile_flag_list"]) > 0:
        vis_weights = vis_flag_tiles(
            obs_out, vis_weights, pyfhd_config["ps_tile_flag_list"], logger
        )

    vis_weights, obs_out = vis_weights_update(vis_weights, obs_out, psf, params)

    if pyfhd_config["split_ps_export"]:
        n_iter = 2
        vis_weights_use, bi_use = split_vis_weights(obs_out, vis_weights)
        obs_out["vis_noise"] = vis_noise_calc(
            obs_out, vis_arr, vis_weights, bi_use=bi_use
        )
        uvf_name = ["even", "odd"]
    else:
        n_iter = 1
        uvf_name = [""]
        obs_out["vis_noise"] = vis_noise_calc(obs_out, vis_arr, vis_weights)
        nb = vis_weights.shape[-1]
        bi_use = (np.arange(nb, dtype=np.int64),)
        vis_weights_use = vis_weights

    # Looks like this is set to False by default?
    residual_flag = obs_out["residual"]
    # Since the model is imported by default, dirty_flag is usually True
    dirty_flag = not residual_flag and vis_model_arr is not None
    # Create the healpix dir Path
    healpix_dir = Path(pyfhd_config["output_dir"], "healpix")
    healpix_dir.mkdir(exist_ok=True)
    for iter in range(n_iter):
        # To save on memory, pyfhd moved to saving the healpix cubes in a polarization loop
        # rather than trying to store the entirety of the everything in memory consuming over
        # 100GB of memory
        for pol_i in range(obs["n_pol"]):
            split = vis_model_freq_split(
                obs_out,
                psf,
                params,
                vis_weights_use,
                vis_model_arr,
                vis_arr,
                pol_i,
                pyfhd_config,
                logger,
                uvf_name=uvf_name[iter],
                bi_use=bi_use[iter],
            )
            if dirty_flag:
                residual_flag = False
            else:
                residual_flag = True
            nf_vis = split["obs"]["nf_vis"]
            nf_vis_use = np.zeros(n_freq_use)
            for freq_i in range(n_freq_use):
                nf_vis_use[freq_i] = np.sum(
                    nf_vis[
                        freq_i
                        * pyfhd_config["n_avg"] : (freq_i + 1)
                        * pyfhd_config["n_avg"]
                    ]
                )

            beam_squared_cube = np.zeros([n_freq_use, hpx_inds.size])
            weights_cube = np.zeros([n_freq_use, hpx_inds.size])
            variance_cube = np.zeros([n_freq_use, hpx_inds.size])
            model_cube = np.zeros([n_freq_use, hpx_inds.size])
            dirty_or_res_cube = np.zeros([n_freq_use, hpx_inds.size])

            for freq_i in range(n_freq_use):
                beam_squared_cube[freq_i, :] = healpix_cnv_apply(
                    beam_arr[pol_i, freq_i] * nf_vis_use[freq_i], hpx_cnv
                )
                weights_cube[freq_i, :] = healpix_cnv_apply(
                    split["weights_arr"][freq_i, :, :], hpx_cnv
                )
                variance_cube[freq_i, :] = healpix_cnv_apply(
                    split["variance_arr"][freq_i, :, :], hpx_cnv
                )
                model_cube[freq_i, :] = healpix_cnv_apply(
                    split["model_arr"][freq_i, :, :], hpx_cnv
                )
                dirty_or_res_cube[freq_i, :] = healpix_cnv_apply(
                    split["residual_arr"][freq_i, :, :], hpx_cnv
                )
            healpix_pol_dict = {
                "obs": split["obs"],
                "hpx_inds": hpx_inds,
                "nside": nside,
                "n_avg": pyfhd_config["n_avg"],
                "beam_squared_cube": beam_squared_cube,
                "weights_cube": weights_cube,
                "variance_cube": variance_cube,
                "model_cube": model_cube,
            }
            if residual_flag:
                healpix_pol_dict["res_cube"] = dirty_or_res_cube
            elif dirty_flag:
                healpix_pol_dict["dirty_cube"] = dirty_or_res_cube
            save(
                healpix_dir
                / f"{pyfhd_config['obs_id']}_{cube_name[iter]}_{obs['pol_names'][pol_i]}.h5",
                healpix_pol_dict,
                f"{pyfhd_config['obs_id']}_{cube_name[iter]}_{obs['pol_names'][pol_i]}",
                logger=logger,
            )

import numpy as np
from numpy.typing import NDArray, ArrayLike
from logging import Logger
from pyfhd.pyfhd_tools.pyfhd_utils import idl_median, histogram


def vis_flag_tiles(
    obs: dict,
    vis_weight_arr: NDArray[np.float64],
    tiles_to_flag: ArrayLike,
    logger: Logger,
) -> np.ndarray:
    """
    Flag tiles in the visibility weights array with a given array or list of tiles to flag
    containing the names of the tiles, NOT the indexes.

    Parameters
    ----------
    obs : dict
        The observation metadata dictionary
    vis_weight_arr : NDArray[np.float64]
        The visibility weight array
    tiles_to_flag : ArrayLike
        The tiles to flag
    logger : Logger
        pyfhd's Logger

    Returns
    -------
    flagged_vis_weight_arr: NDArray[np.float64]
        A vis_weight_arr where the tiles to flag have been set to 0
    """
    tile_flag_list_use = np.array([], dtype=np.int64)
    for flag in tiles_to_flag:
        if type(flag) == str and flag.strip() in obs["baseline_info"]["tile_names"]:
            logger.info(f"Manually flagging tile {flag}")
            flag_idx = np.where(obs["baseline_info"]["tile_names"] == flag.strip())[0][
                0
            ]
            tile_flag_list_use.append(flag_idx + 1)
        else:
            logger.warning(
                f"{flag} wasn't found in obs['baseline_info']['tile_names'], skipping it"
            )
    hist_a, _, ra = histogram(obs["baseline_info"]["tile_a"], min=1)
    hist_b, _, rb = histogram(obs["baseline_info"]["tile_b"], min=1)
    hist_c, _, _ = histogram(tile_flag_list_use, min=1)
    # hist_A and hist_b should be the same size
    hist_ab = hist_a + hist_b
    n_bin = min(np.size(hist_ab), np.size(hist_c))
    tile_cut_i = np.where((hist_ab[0:n_bin] > 0) | (hist_c[0:n_bin] > 0))[0]
    if np.size(tile_cut_i) > 0:
        for cut_idx in range(np.size(tile_cut_i)):
            ti = tile_cut_i[cut_idx]
            na = ra[ra[ti + 1] - 1] - ra[ra[ti]]
            if na > 0:
                vis_weight_arr[:, :, ra[ra[ti] : ra[ti + 1] - 1]] = 0
            nb = rb[rb[ti + 1] - 1] - rb[rb[ti]]
            if nb > 0:
                vis_weight_arr[:, :, rb[rb[ti] : rb[ti + 1] - 1]] = 0
    return vis_weight_arr


def vis_flag_basic(
    vis_weight_arr: NDArray[np.float64],
    vis_arr: NDArray[np.complex128],
    obs: dict,
    pyfhd_config: dict,
    logger: Logger,
) -> tuple[np.ndarray, dict]:
    """
    Do some basic flagging on frequencies and tiles based on the confgiruation given by pyfhd_config
    such as `flag_freq_start`, `flag_freq_end`, `instrument` and `flag_tile_names`. To flag the frequencies and
    tiles, the arrays in `obs['baseline_info']`, *`freq_use`* and *`tile_use`* will be adjusted to 0's where
    the tile or frequency is flagged. The `vis_weight_arr` will be turned to 0's in the associated frequencies
    and tiles that have been flagged.

    Parameters
    ----------
    vis_weight_arr : NDArray[np.float64]
        The visibility weights array
    vis_arr : NDArray[np.complex128]
        The visibilities array
    obs : dict
        The observation dictionary containing the frequency and tile flag arrays
    params : dict
        _description_
    pyfhd_config : dict
        pyfhd's configuration dictionary
    logger : Logger
        pyfhd's logger

    Returns
    -------
    vis_weight_arr : NDArray[np.complex128]
        Updated visibility weights with flagged tiles and frequencies
    obs : dict
        observation metadata dictionary containing updated frequency and tile flags
    """
    freq_arr = obs["baseline_info"]["freq"].copy()

    # If you wish to adjust things based on the configuration option mask_mirror_indices do that here
    # and add the config the pyfhd_setup. There was no description for it in FHD.

    if pyfhd_config["flag_freq_start"]:
        logger.info(
            f'Flagging frequencies less than {pyfhd_config["flag_freq_start"]}MHz'
        )
        frequency_MHz = freq_arr / 1e6
        freq_start_cut = np.where(frequency_MHz < pyfhd_config["flag_freq_start"])
        if np.size(freq_start_cut) > 0:
            vis_weight_arr[:, freq_start_cut, :] = 0
    if pyfhd_config["flag_freq_end"]:
        logger.info(
            f'Flagging frequencies more than {pyfhd_config["flag_freq_end"]}MHz'
        )
        frequency_MHz = freq_arr / 1e6
        freq_end_cut = np.where(frequency_MHz > pyfhd_config["flag_freq_end"])
        if np.size(freq_end_cut) > 0:
            vis_weight_arr[:, freq_end_cut, :] = 0
    # This section replaces the function vis_flag_tile
    if len(pyfhd_config["flag_tiles"]) > 0:
        vis_weight_arr = vis_flag_tiles(
            obs, vis_weight_arr, pyfhd_config["flag_tiles"], logger
        )
    # Here I'm going to assume the mwa data you're using is more than 32 tiles
    # If you wish to implement flagging for mwa when it had 32 tiles, do that here
    # Flagging based on channel width
    if pyfhd_config["flag_frequencies"]:
        freq_avg = 768 // obs["n_freq"]
        channel_edge_flag_width = np.ceil(2 / freq_avg)
        coarse_channel_width = 32 // freq_avg
        fine_channel_i = np.arange(obs["n_freq"]) % coarse_channel_width
        channel_edge_flag = np.where(
            np.minimum(fine_channel_i, (coarse_channel_width - 1) - fine_channel_i)
            < channel_edge_flag_width
        )
        if np.size(channel_edge_flag) > 0:
            vis_weight_arr[:, channel_edge_flag, :] = 0

    tile_a_i = obs["baseline_info"]["tile_a"] - 1
    tile_b_i = obs["baseline_info"]["tile_b"] - 1
    freq_use = np.ones(obs["n_freq"], dtype=np.int64)
    tile_use = np.ones(obs["n_tile"], dtype=np.int64)
    for pol_i in range(obs["n_pol"]):
        baseline_flag = np.max(vis_weight_arr[pol_i], axis=0)
        freq_flag = np.max(vis_weight_arr[pol_i], axis=1)
        fi_use = np.where(freq_flag > 0)
        bi_use = np.where(baseline_flag > 0)

        freq_use_temp = np.zeros(obs["n_freq"], dtype=np.int64)
        if np.size(fi_use) > 0:
            freq_use_temp[fi_use] = 1
        freq_use *= freq_use_temp

        tile_use_temp = np.zeros(obs["n_tile"], dtype=np.int64)
        if np.size(bi_use) > 0:
            tile_use_temp[tile_a_i[bi_use]] = 1
            tile_use_temp[tile_b_i[bi_use]] = 1
        tile_use *= tile_use_temp

    # In the case we choose to not flag any frequencies
    # if pre-processing has flagged frequencies, need to unflag them if the data are nonzero
    # (but DON'T unflag tiles that should be flagged)
    if not pyfhd_config["flag_frequencies"]:
        freq_cut_i = np.where(freq_use == 0)
        if np.size(freq_cut_i) > 0:
            for pol_i in range(obs["n_pol"]):
                freq_flag = np.maximum(np.max(vis_weight_arr[pol_i], axis=-1), 0)
                freq_flag = np.minimum(freq_flag, 1)
                freq_unflag_i = np.where(freq_flag == 0)[0]
                if np.size(freq_unflag_i) > 0:
                    baseline_flag = np.maximum(np.max(vis_weight_arr[pol_i], axis=0), 0)
                    bi_use = np.where(baseline_flag > 0)[0]
                    for fi in range(np.size(freq_unflag_i)):
                        data_test = np.abs(vis_arr[pol_i][freq_unflag_i[fi], bi_use])
                        data_test = data_test.T
                        unflag_i = np.where(data_test > 0)
                        if np.size(unflag_i) > 0:
                            vis_weight_arr[pol_i][
                                freq_unflag_i[fi], bi_use[unflag_i]
                            ] = 1
        freq_use = np.ones(obs["n_freq"], dtype=np.int64)

    # Time based flagging
    if np.min(obs["baseline_info"]["time_use"]) <= 0:
        bin_offset = obs["baseline_info"]["bin_offset"]
        bin_offset = np.hstack([bin_offset, np.size(obs["baseline_info"]["tile_a"])])
        time_bin = np.zeros(np.size(obs["baseline_info"]["tile_a"]))
        for ti in range(obs["n_time"]):
            if obs["baseline_info"]["time_use"][ti] <= 0:
                vis_weight_arr[:, :, bin_offset[ti] : bin_offset[ti + 1] - 1] = 0

    tile_use_indexes = np.where((tile_use) & (obs["baseline_info"]["tile_use"]))[0]
    tile_use_zeros = np.zeros_like(obs["baseline_info"]["tile_use"])
    tile_use_zeros[tile_use_indexes] = 1
    obs["baseline_info"]["tile_use"] = tile_use_zeros
    freq_use_indexes = np.where((freq_use) & (obs["baseline_info"]["freq_use"]))[0]
    freq_use_zeros = np.zeros_like(obs["baseline_info"]["freq_use"])
    freq_use_zeros[freq_use_indexes] = 1
    obs["baseline_info"]["freq_use"] = freq_use_zeros

    obs["n_time_flag"] = np.count_nonzero(obs["baseline_info"]["time_use"] == 0)
    obs["n_tile_flag"] = np.count_nonzero(obs["baseline_info"]["tile_use"] == 0)
    obs["n_freq_flag"] = np.count_nonzero(obs["baseline_info"]["freq_use"] == 0)

    if np.max(vis_weight_arr) == 0:
        raise ValueError("All data has been flagged, check your flagging parameters")

    return vis_weight_arr, obs


def vis_flag(
    vis_arr: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    obs: dict,
    params: dict,
    logger: Logger,
) -> tuple[np.ndarray, dict]:
    """
    Flag frequencies, times, and tiles which are 3 sigma away from the median
    of the visibility amplitudes, along with baselines which are outside the
    supplied minimum and maximum baseline distances.

    Parameters
    ----------
    vis_arr : NDArray[np.complex128]
        The visibility array
    vis_weights : NDArray[np.float64]
        The visibility weights array
    obs : dict
        The observation dictionary
    params : dict
        The dictionary containing uu, vv, ww
    logger : Logger
        pyfhd's Logger

    Returns
    -------
    vis_weights : np.ndarray
        Flagged vis_weights
    obs : dict
        Flagged tiles and frequencies observation metadata dictionary
    """
    flag_nsigma = 3
    data_abs = np.abs(vis_arr[0])
    if obs["n_pol"] > 1:
        data_abs = np.sqrt(data_abs**2 + np.abs(vis_arr[1]) ** 2)
    n_tiles_use = max(
        np.max(obs["baseline_info"]["tile_a"]), np.max(obs["baseline_info"]["tile_b"])
    )
    uv_dist = np.sqrt(params["uu"] ** 2 + params["vv"] ** 2) * idl_median(
        obs["baseline_info"]["freq"]
    )
    cut_baselines_i = np.where(
        (uv_dist < obs["min_baseline"]) | (uv_dist > obs["max_baseline"])
    )[0]
    if cut_baselines_i.size > 0:
        vis_weights[: obs["n_pol"], :, cut_baselines_i] = 0

    tile_fom = np.zeros(n_tiles_use, dtype=np.float64)
    for tile_i in range(n_tiles_use):
        tile_ai = np.where((obs["baseline_info"]["tile_a"] - 1) == tile_i)
        tile_bi = np.where((obs["baseline_info"]["tile_b"] - 1) == tile_i)
        if tile_ai[0].size == 0 and tile_bi[0].size == 0:
            continue
        if tile_bi[0].size > 0:
            if tile_ai[0].size > 0:
                tile_abi = np.hstack([tile_ai[0], tile_bi[0]])
            else:
                tile_abi = tile_bi
        else:
            tile_abi = tile_ai
        data_subset = data_abs[:, tile_abi]
        for pol_i in range(min(obs["n_pol"], 2)):
            # Dunno why this needs two separate indexes but it works so leave it
            i_use = np.where((vis_weights[pol_i][:, tile_abi] > 0) & (data_subset > 0))
            if i_use[0].size > 10:
                tile_fom[tile_i] += np.std(data_subset[i_use])

    freq_fom = np.zeros(obs["n_freq"])
    for freq_i in range(obs["n_freq"]):
        data_subset = data_abs[freq_i, :]
        for pol_i in range(min(obs["n_pol"], 2)):
            i_use = np.where((vis_weights[pol_i, freq_i, :] > 0) & (data_subset > 0))
            if i_use[0].size > 10:
                freq_fom[freq_i] += np.std(data_subset[i_use])

    freq_nonzero = np.nonzero(freq_fom)[0]
    tile_nonzero = np.nonzero(tile_fom)
    tile_mean = idl_median(tile_fom[tile_nonzero])
    tile_dev = np.std(tile_fom[tile_nonzero])
    # IDL Median with WIDTH set is doing a median_filter with the size indicating the kernel size
    freq_mean1 = idl_median(freq_fom[freq_nonzero], width=obs["n_freq"] // 20)
    freq_mean = np.zeros(obs["n_freq"])
    freq_mean[freq_nonzero] = freq_mean1
    freq_dev = np.std(freq_fom[freq_nonzero] - freq_mean[freq_nonzero])

    # We actually want the complements of the where from IDL translation, adjusted conditions accordingly
    tile_cut0 = np.where(
        (np.abs(tile_mean - tile_fom) <= 2 * flag_nsigma * tile_dev) | (tile_fom != 0)
    )[0]
    freq_cut0 = np.where(
        (np.abs(freq_mean - freq_fom) <= 2 * flag_nsigma * freq_dev) | (freq_fom != 0)
    )[0]
    tile_mean2 = idl_median(tile_fom[tile_cut0])
    tile_dev2 = np.std(tile_fom[tile_cut0])
    freq_mean2 = idl_median(freq_fom, width=obs["n_freq"] // 20)
    freq_dev2 = np.std((freq_fom - freq_mean2)[freq_cut0])
    # Currently assuming tile_cut and freq_cut are 1D
    tile_cut = np.where(
        (np.abs(tile_mean2 - tile_fom) > (flag_nsigma * tile_dev2)) | (tile_fom == 0)
    )[0]
    freq_cut = np.where(
        (np.abs(freq_mean2 - freq_fom) > (flag_nsigma * freq_dev2)) | (freq_fom == 0)
    )[0]

    if tile_cut.size > 0:
        logger.info(f"Tiles Cut: {obs['baseline_info']['tile_names'][tile_cut]}")
        for bad_i in range(tile_cut.size):
            cut_a_i = np.where(obs["baseline_info"]["tile_a"] == tile_cut[bad_i] + 1)
            cut_b_i = np.where(obs["baseline_info"]["tile_b"] == tile_cut[bad_i] + 1)
            if cut_a_i[0].size > 0:
                vis_weights[0 : obs["n_pol"], :, cut_a_i] = 0
            if cut_b_i[0].size > 0:
                vis_weights[0 : obs["n_pol"], :, cut_b_i] = 0
        obs["baseline_info"]["tile_use"][tile_cut] = 0
    if freq_cut.size > 0:
        vis_weights[0 : obs["n_pol"], freq_cut] = 0
        obs["baseline_info"]["freq_use"][freq_cut] = 0

    bin_offset = np.append(
        obs["baseline_info"]["bin_offset"], obs["baseline_info"]["tile_a"].size
    )
    time_bin = np.zeros(obs["baseline_info"]["tile_a"].size)
    for ti in range(len(obs["baseline_info"]["time_use"])):
        time_bin[bin_offset[ti] : bin_offset[ti + 1]] = ti
    time_fom = np.zeros(obs["baseline_info"]["time_use"].size)
    for ti in range(len(obs["baseline_info"]["time_use"])):
        data_subset = data_abs[:, bin_offset[ti] : bin_offset[ti + 1]]
        for pol_i in range(min(obs["n_pol"], 2)):
            i_use = np.where(
                vis_weights[pol_i, :, bin_offset[ti] : bin_offset[ti + 1]] > 0
            )
            if i_use[0].size > 10:
                time_fom[ti] += np.std(data_subset[i_use])
    time_nonzero = np.nonzero(time_fom)
    time_mean = idl_median(time_fom[time_nonzero])
    time_dev = np.std(time_fom[time_nonzero])
    time_cut0 = np.where(
        (np.abs(time_mean - time_fom) <= 2 * flag_nsigma * time_dev) | (time_fom != 0)
    )
    time_mean2 = idl_median(time_fom[time_cut0])
    time_dev2 = np.std(time_fom[time_cut0])
    time_cut = np.where(
        (np.abs(time_mean2 - time_fom) > 2 * flag_nsigma * time_dev2) | (time_fom == 0)
    )[0]
    for ti in range(time_cut.size):
        ti_cut = np.where(time_bin == time_cut[ti])
        if ti_cut.size > 0:
            vis_weights[: obs["n_pol"], :, ti_cut] = 0

    obs["n_vis"] = (np.where(vis_weights[0] > 0)[0]).size

    return vis_weights, obs

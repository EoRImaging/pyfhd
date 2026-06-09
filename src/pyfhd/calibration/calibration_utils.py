import numpy as np
from numpy.typing import NDArray
from logging import Logger
from pyfhd.pyfhd_tools.pyfhd_utils import (
    resistant_mean,
    weight_invert,
    rebin,
    histogram,
)
from copy import deepcopy
from astropy.io import fits
from astropy.constants import c
from pathlib import Path
from scipy.ndimage import uniform_filter
import importlib_resources


def vis_extract_autocorr(
    obs: dict,
    vis_arr: NDArray[np.complex128],
    pyfhd_config: dict,
    auto_tile_i: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.integer]]:
    """
    Extract the auto-correlations if they exist from the full visibility array.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    vis_arr : NDArray[np.complex128]
        Uncalibrated data visiblities
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    auto_tile_i : NDArray[np.integer] | None, optional
        Index array for auto-correlation visibilities, by default None

    Returns
    -------
    autocorr: NDArray[np.float64]
        The auto-correlation visibilities
    auto_tile_i : NDArray[np.integer]
        The index array for auto-correlation visibilities
    """

    autocorr_i = np.where(
        obs["baseline_info"]["tile_a"] == obs["baseline_info"]["tile_b"]
    )[0]
    if autocorr_i.size > 0:
        auto_tile_i = obs["baseline_info"]["tile_a"][autocorr_i] - 1
        # As auto_tile_i is used for indexing we need to make it an integer array
        auto_tile_i = auto_tile_i.astype(int)
        auto_tile_i_single = np.unique(auto_tile_i)
        # expect it as a list of 2D arrays, so there might be trouble
        if not pyfhd_config["cal_time_average"]:
            freq_tile_shape = np.real(vis_arr[0][:, autocorr_i]).shape
            auto_corr = np.zeros([obs["n_pol"]] + list(freq_tile_shape))
        else:
            auto_corr = np.zeros((obs["n_pol"], obs["n_freq"], auto_tile_i_single.size))

        for pol_i in range(obs["n_pol"]):
            # Auto-correlations by definition are enirely real, so take the real part here
            # index this way in case people have vis_arr as one dimensional
            # array, containing two further 2D arrays, rather than a proper 3D
            # array. Turns out this indexing is consistent across both cases
            auto_vals = np.real(vis_arr[pol_i][:, autocorr_i])
            if pyfhd_config["cal_time_average"]:
                auto_single = np.zeros((obs["n_freq"], auto_tile_i_single.size))
                time_inds = np.where(obs["baseline_info"]["time_use"])[0]
                for tile_i in range(auto_tile_i_single.size):
                    baseline_i = np.where(auto_tile_i == auto_tile_i_single[tile_i])[0]
                    baseline_i = baseline_i[time_inds]
                    if time_inds.size > 1:
                        # OK, auto_vals is of shape (n_freq, n_autos)
                        # We want to average a subset of baseline_i within auto_vals,
                        # to average a specific auto-correlation over time
                        auto_single[:, tile_i] = (
                            np.sum(
                                auto_vals[:, baseline_i][np.arange(obs["n_freq"]), :],
                                axis=1,
                            )
                            / time_inds.size
                        )

                    else:
                        auto_single[:, tile_i] = auto_vals[:, baseline_i][
                            np.arange(obs["n_freq"]), :
                        ]
                auto_vals = auto_single
            auto_corr[pol_i, :, :] = auto_vals
        if pyfhd_config["cal_time_average"]:
            auto_tile_i = auto_tile_i_single
        return auto_corr, auto_tile_i
    else:
        # Return auto_corr as 0 and auto_tile_i as an empty array
        return np.zeros(1), np.zeros(0)


def vis_cal_auto_init(
    obs: dict,
    cal: dict,
    vis_arr: NDArray[np.complex128],
    vis_model_arr: NDArray[np.complex128],
    vis_auto: NDArray[np.float64],
    vis_auto_model: NDArray[np.float64],
    auto_tile_i: NDArray[np.integer],
) -> NDArray[np.complex128]:
    """
    Initialize the calibration solutions using the autocorrelations prior to the linear least squares fit
    for faster convergence. The auto-correlations and cross-correlations have separate formulisms for the
    initialization.

    Autos -- Square root of "Ratio of data autos to model autos" x "overall mean for both crosses and
    autos of ratio of data to model" / "overall mean of ratio of data autos to model autos, with 1's
    for crosses", results in a per-frequency, per-baseline gain scaling.
    Crosses -- Square root of "overall mean for both crosses and autos of ratio of data to model" /
    "overall mean of ratio of data autos to model autos, with 1's for crosses", results in a single
    number gain scaling.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    vis_arr : NDArray[np.complex128]
        Uncalibrated data visiblities
    vis_model_arr : NDArray[np.complex128]
        Simulated model visibilites
    vis_auto : NDArray[np.float64]
        Data auto-correlations
    vis_auto_model : NDArray[np.float64]
        Simulated model auto-correlations
    auto_tile_i : NDArray[np.integer]
        Index array for auto-correlation visibilities

    Returns
    -------
    auto_gain : NDArray[np.complex128]
        Initialization gain array
    """
    # Set out the
    auto_scale = np.zeros((cal["n_pol"], obs["n_freq"], obs["n_tile"]))
    auto_gain = np.ones(
        (cal["n_pol"], obs["n_freq"], obs["n_tile"]), dtype=np.complex128
    )
    freq_i_use = np.where(obs["baseline_info"]["freq_use"])[0]
    for pol_i in range(cal["n_pol"]):
        res_mean_data = resistant_mean(np.abs(vis_arr[pol_i, freq_i_use, :]), 2)
        res_mean_model = resistant_mean(np.abs(vis_model_arr[pol_i, freq_i_use, :]), 2)
        auto_scale[pol_i] = np.sqrt(res_mean_data / res_mean_model)
    auto_gain = np.ones(
        (cal["n_pol"], obs["n_freq"], obs["n_tile"]), dtype=np.complex128
    )
    # Would love to get rid of the loop altogether, haven't figurted it out yet
    for pol_i in range(cal["n_pol"]):
        auto_gain[pol_i, :, :][:, auto_tile_i] = np.sqrt(
            vis_auto[pol_i, :, :] * weight_invert(vis_auto_model[pol_i, :, :])
        )
        auto_gain[pol_i, :, :] *= auto_scale[pol_i] * weight_invert(
            np.mean(auto_gain[pol_i, :, :])
        )
        auto_gain[pol_i, :, :][np.isnan(auto_gain[pol_i, :, :])] = 1
        auto_gain[pol_i, :, :][auto_gain[pol_i, :, :] <= 0] = 1
    return auto_gain


def vis_calibration_flag(
    obs: dict, cal: dict, pyfhd_config: dict, logger: Logger
) -> dict:
    """
    Flag tile and frequency outliers based on the calibration solutions. First, iteratively flag a maximum of three
    times on amplitude with three tests: 1) flag frequencies above 5 sigma, 2) flag tiles above 5 sigma, and 3) flag
    tiles either 2x lower or 2x higher than average. Second, iteratively flag a maximum of three times on phase with
    two tests: 1) flag tiles with slopes above 5 sigma, and 2) flag tiles with per-frequency deviations from their slope
    above 5 sigma.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger for displaying errors and info to the log files

    Returns
    -------
    obs: dict
        Observation metadata dictionary
    """
    amp_sigma_threshold = 5
    amp_threshold = 2
    phase_sigma_threshold = 5
    freq_use = obs["baseline_info"]["freq_use"].copy()
    for pol_i in range(cal["n_pol"]):
        tile_use_i = np.nonzero(obs["baseline_info"]["tile_use"])[0]
        freq_use_i = np.nonzero(freq_use)[0]
        gain = cal["gain"][pol_i]
        phase = np.arctan2(gain.imag, gain.real)
        amp = np.abs(gain)

        # first flag based on overall amplitude
        # extract_subarray is not being used as it was FHD's way of taking the fact that
        # IDL's indexing can be weird and won't allow to index the result
        # use np.where on gain, which will be multidimensional as well
        amp_sub = amp[:, tile_use_i][freq_use_i, :]
        gain_freq_fom = np.std(amp_sub, axis=1)
        # Calculate the y values from a polynomial fit and keep the standard deviation of the error in gain_tile_fom
        amp_sub_fit = np.zeros_like(amp_sub)
        # Polynomial polyval can only take 1d arrays and doesn't vectorize across a 2d array.
        for tile_i in range(tile_use_i.size):
            amp_sub_fit[:, tile_i] = np.polynomial.polynomial.polyval(
                freq_use_i,
                np.polynomial.Polynomial.fit(
                    freq_use_i,
                    amp_sub[:, tile_i],
                    deg=pyfhd_config["cal_amp_degree_fit"],
                )
                .convert()
                .coef,
            )
        # Get standard deviation of the residuals on a per tile basis
        gain_tile_fom = np.std(amp_sub - amp_sub_fit, axis=0)
        # Get the median of every tile, why is it call avg?
        gain_tile_avg = np.median(amp_sub, axis=0)
        gain_freq_fom[np.isnan(gain_freq_fom)] = 0
        gain_tile_fom[np.isnan(gain_tile_fom)] = 0
        freq_cut_i = np.where(gain_freq_fom == 0)[0]
        freq_uncut_i = np.nonzero(gain_freq_fom)[0]
        if freq_cut_i.size > 0:
            freq_use[freq_use_i][freq_cut_i] = 0
        tile_cut_i = np.where(gain_tile_fom == 0)[0]
        tile_uncut_i = np.nonzero(gain_tile_fom)[0]
        if tile_cut_i.size > 0:
            obs["baseline_info"]["tile_use"][tile_use_i][tile_cut_i] = 0
        if freq_uncut_i.size == 0 or tile_uncut_i.size == 0:
            logger.error(
                "The frequency and tile flagging inside calibration found some values not detected in previous flagging or calibration"
            )

        n_addl_cut = max(freq_cut_i.size + tile_cut_i.size, 1)
        n_cut = freq_cut_i.size + tile_cut_i.size
        iter = 0
        while n_addl_cut > 0 and iter < 3:
            gain_freq_sigma = np.std(gain_freq_fom[freq_uncut_i])
            gain_tile_sigma = np.std(gain_tile_fom[tile_uncut_i])
            freq_cut_i = np.where(
                (
                    gain_freq_fom
                    - np.median(gain_freq_fom[freq_uncut_i])
                    - amp_sigma_threshold * gain_freq_sigma
                )
                > 0
            )[0]
            # Update the complement of freq_cut_i (the NOT) i.e. freq_uncut_i
            freq_uncut_i = freq_uncut_i[~np.isin(freq_uncut_i, freq_cut_i)]
            tile_cut_test1 = (
                gain_tile_fom
                - np.median(gain_tile_fom[tile_uncut_i])
                - amp_sigma_threshold * gain_tile_sigma
            ) > 0
            tile_cut_test2 = (
                gain_tile_avg < np.median(gain_tile_avg) / amp_threshold
            ) | (gain_tile_avg > np.median(gain_tile_avg) * amp_threshold)
            tile_cut_i = np.where(tile_cut_test1 | tile_cut_test2)[0]
            # Update the complement of tile_cut_i (the NOT) i.e. tile_uncut_i
            tile_uncut_i = tile_uncut_i[~np.isin(tile_uncut_i, tile_cut_i)]
            n_addl_cut = (freq_cut_i.size + tile_cut_i.size) - n_cut
            n_cut = freq_cut_i.size + tile_cut_i.size
            iter += 1
        if (freq_cut_i.size) > 0:
            freq_use[freq_use_i[freq_cut_i]] = 0
        if (tile_cut_i.size) > 0:
            obs["baseline_info"]["tile_use"][tile_use_i[tile_cut_i]] = 0

        # Reset freq_use_i and tile_use_i for flagging based on phase
        tile_use_i = np.nonzero(obs["baseline_info"]["tile_use"])[0]
        freq_use_i = np.nonzero(freq_use)[0]

        # Start flagging based on phase
        phase_sub = phase[:, tile_use_i][freq_use_i, :]
        phase_slope_arr = np.empty(tile_use_i.size)
        phase_sigma_arr = np.empty(tile_use_i.size)
        for tile_i in range(tile_use_i.size):
            phase_use = np.unwrap(phase_sub[:, tile_i])
            phase_params = np.polynomial.polynomial.Polynomial.fit(
                freq_use_i, phase_use, deg=pyfhd_config["cal_phase_degree_fit"]
            )
            phase_params = phase_params.convert().coef
            phase_fit = np.polynomial.polynomial.polyval(freq_use_i, phase_params)
            phase_sigma2 = np.std(phase_use - phase_fit)
            # In an unusual scenario sometimes you'll get an all 0 array to fit on, which gives only a zero back for the fit
            phase_slope_arr[tile_i] = phase_params[1] if phase_params.size > 1 else 0
            phase_sigma_arr[tile_i] = phase_sigma2
        iter = 0
        n_addl_cut = 1
        n_cut = 0
        while n_addl_cut > 0 and iter < 3:
            slope_sigma = np.nanstd(phase_slope_arr)
            tile_cut_test1 = (
                np.abs(phase_slope_arr) - np.median(np.abs(phase_slope_arr))
            ) > phase_sigma_threshold * slope_sigma
            tile_cut_test2 = (phase_sigma_arr - np.median(phase_sigma_arr)) > (
                phase_sigma_threshold * np.nanstd(phase_sigma_arr)
            )
            tile_cut_i = np.where(tile_cut_test1 | tile_cut_test2)[0]
            n_addl_cut = tile_cut_i.size - n_cut
            n_cut = tile_cut_i.size
            iter += 1
        if tile_cut_i.size > 0:
            obs["baseline_info"]["tile_use"][tile_use_i[tile_cut_i]] = 0
    # If we decide not to flag the frequencies, ignore any frequency flagging
    if pyfhd_config["flag_calibration_frequencies"]:
        obs["baseline_info"]["freq_use"] = freq_use
    # Return the obs with an updated baseline_info on the use of tiles and frequency
    return obs


def transfer_bandpass(
    obs: dict, cal: dict, pyfhd_config: dict, logger: Logger
) -> tuple[dict, dict]:
    """
    Apply a previously saved bandpass via a calfits file (github:pyuvdata). Check adherance to standards,
    and match the polarizations, frequencies, timing, pointings, and tiles.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger for displaying errors and info to the log files

    Returns
    -------
    cal_bandpass : dict
        Calibration dictionary with bandpass gains
    cal_remainder : dict
        calibration dictionary with residuals after removing the bandpass gains
    """
    cal_bandpass = {}
    try:
        calfits = fits.open(
            Path(pyfhd_config["input_path"], pyfhd_config["cal_bp_transfer"])
        )
        # Get the data
        data_array = calfits[0].data
        # Read in the header
        naxis = calfits[0].header["naxis"]
        n_jones = calfits[0].header["njones"]
        if "delay" in calfits[0].header["caltype"]:
            raise RuntimeWarning(
                "Input Delay calibration not supported at this time, skipping calibration bandpass transfer."
            )
        time_integration = calfits[0].header["inttime"]
        freq_channel_width = calfits[0].header["chwidth"]
        x_orient = calfits[0].header["xorient"].strip()
        data_dims = np.empty(naxis, dtype=np.integer)
        data_types = naxis * [""]
        for naxis_i in range(naxis):
            # Get the dimensions of the data
            data_dims[naxis_i] = calfits[0].header[f"naxis{naxis_i + 1}"]
            # Get the ctypes
            data_types[naxis_i] = (
                calfits[0].header[f"ctype{naxis_i + 1}"].lower().strip()
            )
        data_types = np.array(data_types)
        # Get the indexes for the FITS standard checks
        data_index = np.nonzero("narrays" == data_types)[0][0]
        ant_index = np.nonzero("antaxis" == data_types)[0][0]
        freq_index = np.nonzero("freqs" == data_types)[0][0]
        time_index = np.nonzero("time" == data_types)[0][0]
        jones_index = np.nonzero("jones" == data_types)[0][0]
        # Deal with spec_wind_index separately as default highband file doesn't have this in
        # May cause issues with other fits files, adjust the code then.
        spec_wind_index = np.nonzero("if" == data_types)[0]
        if spec_wind_index.size == 0:
            spec_wind_index = -1

        # Check the indexes given to see if they match standards, if not raise exception
        if (
            data_index != 0
            or ant_index != 4
            or freq_index != 3
            or time_index != 2
            or jones_index != 1
        ):
            if (
                data_index == 0
                and ant_index == 5
                and freq_index == 3
                and time_index == 2
                and jones_index == 1
                and spec_wind_index == 4
            ):
                logger.info("Calfits adheres to the Fall 2018 pyuvdata convention")
                if calfits[0].header["naxis5"] != 1:
                    raise RuntimeWarning(
                        "Calfits file includes more than one spectral window. Note that this feature is not yet supported in pyfhd."
                    )
                # Remove spectral window dimension for compatibility
                data_array = np.mean(data_array, axis=0)
            else:
                raise RuntimeWarning(
                    "Calfits file does not appear to adhere to standard. Please see github:pyuvdata/docs/references"
                )

        freq_start = calfits[0].header[f"crval{freq_index + 1}"]
        time_start = calfits[0].header[f"crval{time_index + 1}"]
        time_delt = calfits[0].header[f"cdelt{time_index + 1}"]
        jones_start = calfits[0].header[f"crval{jones_index + 1}"]
        jones_delt = calfits[0].header[f"cdelt{jones_index + 1}"]

        n_ant_data = data_array.shape[0]
        n_freq = data_array.shape[1]
        n_time = data_array.shape[2]

        # Check whether the number of polarizations specified matches the observation analysis run
        jones_type_matrix = np.zeros(data_dims[1])
        for jones_i in range(1, data_dims[1]):
            jones_type_matrix[jones_i - 1] = jones_start + (jones_delt * jones_i)
        if data_dims[1] > obs["n_pol"]:
            logger.warning(
                "More polarizations in calibration fits file than in observation analysis. Reducing calibration to match obs."
            )
            data_dims[1] = obs["n_pol"]
            jones_type_matrix = jones_type_matrix[0 : obs["n_pol"]]
            data_array = data_array[:, :, :, 0 : obs["n_pol"], :]
        elif data_dims[1] < obs["n_pol"]:
            raise RuntimeWarning(
                "Not enough polarizations defined in the calibration fits file."
            )

        # Switch the pol convention to FHD standard if necessary
        if x_orient == "north" or x_orient == "south":
            data_array[:, :, :, 0, :], data_array[:, :, :, 1, :] = (
                data_array[:, :, :, 1, :],
                data_array[:, :, :, 0, :],
            )
            if obs["n_pol"] > 2:
                data_array[:, :, :, 2, :], data_array[:, :, :, 3, :] = (
                    data_array[:, :, :, 3, :],
                    data_array[:, :, :, 2, :],
                )

        # Check to see if the calibraton and observation frequency resolution match
        if freq_channel_width != obs["freq_res"]:
            freq_factor = obs["freq_res"] / freq_channel_width
            if freq_factor >= 1:
                logic_test = freq_factor - np.floor(freq_factor)
            else:
                logic_test = 1 / freq_factor - np.floor(1 / freq_factor)
            if logic_test != 0:
                raise RuntimeWarning(
                    f"Calfits input freq channel width is not easily castable to the observation, different by a factor of {freq_factor}"
                )
            if freq_start != obs["baseline_info"]["freq"][0]:
                raise RuntimeWarning(
                    "Calfits input freq start is not equal to observation freq start"
                )
            # Downselect the data array
            if freq_factor > 1:
                logger.warning(
                    f"Calfits input freq channel width is different by a factor of {freq_factor}. Avergaing Down."
                )
                # Set flagged indices to NAN to remove them from mean calculation
                flag_inds = np.where(np.abs(np.squeeze(data_array[:, :, :, :, 2])) == 1)
                data_array[flag_inds][0] = np.nan
                data_array[flag_inds][1] = np.nan
                for channel_i in range(obs["n_freq"]):
                    data_array[:, channel_i, :, :, :] = np.nanmean(
                        data_array[
                            :,
                            max(
                                channel_i * freq_factor - np.floor(freq_factor / 2), 0
                            ) : channel_i * freq_factor + np.floor(freq_factor / 2.0),
                            :,
                            :,
                            :,
                        ]
                    )
                data_array = data_array[:, 0 : obs["n_freq"], :, :, :]
            elif freq_factor < 1:
                logger.warning(
                    f"Calfits input freq channel width is different by a factor of {freq_factor}. Using linear interpolation"
                )
                # The IDL code has 5 nested loops, and I can't think of the vectorization right now in a reasonable ampount of time
                # Someone please vectorize laster if you can
                data_array_temp = np.zeros(
                    (data_dims[4], obs["n_freq"], n_time, n_jones, 2)
                )
                for data_i in range(2):
                    for jones_i in range(n_jones):
                        for times_i in range(n_time):
                            for tile_i in range(data_dims[4]):
                                for channel_i in range(obs["n_freq"] - 1):
                                    start_idx = int(channel_i * (1.0 / freq_factor))
                                    end_idx = int((channel_i + 1) * (1.0 / freq_factor))
                                    data_array_temp[
                                        tile_i,
                                        start_idx:end_idx,
                                        times_i,
                                        jones_i,
                                        data_i,
                                    ] = np.interp(
                                        np.arange(start_idx, end_idx, 1.0),
                                        np.arange(channel_i, channel_i + 1, 1.0),
                                        data_array[
                                            data_i,
                                            jones_i,
                                            times_i,
                                            channel_i : channel_i + 1,
                                            tile_i,
                                        ],
                                    )
                data_array_temp[:, obs["n_freq"] - 1, :, :, :] = data_array[
                    :, n_freq - 1, :, :, :
                ]
                data_array = np.copy(data_array_temp)

        # Check to see what time range this needs to be applied to, and if pointings are necessary
        if n_time != 1:
            sec_upperlimit = 2000
            sec_lowerlimit = 1600
            if (
                time_integration < sec_upperlimit and time_integration > sec_lowerlimit
            ) or (time_delt < sec_upperlimit and time_delt > sec_lowerlimit):
                # Calibration fits are per pointing
                # Keep all the delay patterns in a dictionary
                delay_patterns = {
                    (-5): [0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18],
                    (-4): [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
                    (-3): [0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9],
                    (-2): [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6],
                    (-1): [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                    (0): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    (5): [15, 10, 5, 0, 16, 11, 6, 1, 17, 12, 7, 2, 18, 13, 8, 3],
                    (4): [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3],
                    (3): list(
                        reversed([0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9])
                    ),
                    (2): list(
                        reversed([0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6])
                    ),
                    (1): list(
                        reversed([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
                    ),
                }
                obs_pointing = None
                for pointing, pattern in delay_patterns.items():
                    if np.array_equal(obs["delays"], pattern):
                        obs_pointing = pointing
                if obs_pointing is None:
                    raise RuntimeWarning(
                        "Pointing number not within 5 pointings around zenith."
                    )
                obs_julian_date = obs["astr"]["mjdobs"] + 2400000.5

                # Find corresponding in the calfits
                # Number of days since Auguest 23 2013
                days_since_ref = np.floor(obs_julian_date) - np.floor(time_start)
                # pointing start shift amount depending on how many days since ref
                obs_pointing_shift_since_ref = ((24 - 23.9344699) / 24) * days_since_ref
                # pointing start time for HH:MM:SS on Aug23 (in JD) plus comparible pointing start time for reference, using calculated shift
                pointing_jdhms_ref = (
                    np.array(
                        [
                            0.13611,
                            0.15157,
                            0.17565,
                            0.19963,
                            0.22222,
                            0.24342593,
                            0.26453705,
                            0.28574074,
                            0.30694,
                            0.33657407,
                        ]
                    )
                    + obs_pointing_shift_since_ref
                )
                pointing_num_ref = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
                # pointing start time for HH:MM:SS for calfits day (in JD)
                pointing_jdhms_calfits = time_start - np.floor(time_start)
                # find the closest index between reference and calfits
                pointing_calfits_index = np.argmin(
                    np.abs(pointing_jdhms_ref - pointing_jdhms_calfits)
                )
                # make sure the index is greater than the pointing start time
                if pointing_jdhms_calfits < pointing_jdhms_ref[pointing_calfits_index]:
                    pointing_calfits_index -= 1

                if (
                    pointing_jdhms_calfits
                    > (
                        np.max(pointing_jdhms_ref)
                        + (pointing_jdhms_ref[1] - pointing_jdhms_ref[0])
                    )
                ) or (
                    pointing_jdhms_calfits
                    < (
                        np.min(pointing_jdhms_ref)
                        - (pointing_jdhms_ref[1] - pointing_jdhms_ref[0])
                    )
                ):
                    raise RuntimeWarning(
                        "Calfits does not start between five pointings before zenith and four pointings after zenith. Not suitable for pointing cal at this time."
                    )
                # find which pointing is the start of the calfits data
                pointing_calfits_start = pointing_num_ref[pointing_calfits_index]
                if obs_pointing < pointing_calfits_start:
                    raise RuntimeWarning(
                        "Calfits file does not contain pointing of observation"
                    )
                # select pointing index in calfits that matches observation
                obs_pointing_index = abs(pointing_calfits_start - obs_pointing)
                # choose the corresponding pointing from the calfits data array
                data_array = data_array[:, :, obs_pointing_index, :, :]
            elif np.floor(time_delt) == np.floor(obs["time_res"]):
                # Calibration fits are per-timeres
                logger.info(
                    "Averaging calfits to observation length, an FHD requirement at this time."
                )
                data_array_temp = np.zeros(
                    [data_dims[4], obs.n_freq, 1, data_dims[1], data_dims[0]]
                )
                data_array_temp[:, :, 0, :, :] = np.mean(data_array, axis=2)
                data_array = np.copy(data_array_temp)
            else:
                # Calibration fits are for a random set of times
                logger.info(
                    "Finding closest match in time between calfits and obs. Obs metadata assumed to report start time, calfits metadata assumed to report center time."
                )
                time_delta = time_integration / (60 * 60 * 24)
                time_array = np.full(n_time, time_start + time_delta)
                obs_julian_date = (
                    obs["astr"]["mjdobs"]
                    + 2400000.5
                    + ((obs["n_time"] * obs["time_res"]) / (2 * 60 * 60 * 24))
                )
                if (obs_julian_date < time_array[0] - 2 * time_delta) or (
                    obs_julian_date > time_array[-1] + 2 * time_delta
                ):
                    raise RuntimeWarning(
                        "Observation does not seem to fit within the time frame of the calfits"
                    )
                # find the closest index between calfits and observation
                time_index = np.argmin(np.abs(obs_julian_date - time_array))
                data_array = data_array[:, :, time_index, :, :]

        # Check number of tiles
        if n_ant_data != obs["n_tile"]:
            raise RuntimeWarning(
                "Number of antennas in calfits file does match observation antenna number"
            )

        # Now that the checks are done, return the cal structure
        cal_bandpass["n_pol"] = min(obs["n_pol"], 2)
        cal_bandpass["conv_thresh"] = 1e-7
        cal_bandpass["gain"] = np.full(
            (cal_bandpass["n_pol"], obs["n_tile"], obs["n_freq"]),
            pyfhd_config["cal_gain_init"],
        )
        cal_bandpass["gain"][0 : obs["n_pol"], :, :] = np.squeeze(
            data_array[:, :, 0, 0 : obs["n_pol"], 0]
        ) + 1j * np.squeeze(data_array[:, :, 0, 0 : obs["n_pol"], 1])
        logger.info("Calfits File has been read and cal_bandpass has been created")
    except FileNotFoundError:
        logger.error(
            f"{pyfhd_config['cal_bp_transfer']} file wasn't found, skipping calibration bandpass transfer"
        )
        return {}, {}
    except RuntimeWarning as e:
        logger.error(e)
        return {}, {}
    cal_remainder = deepcopy(cal)
    cal_remainder["gain"][0 : cal["n_pol"], :, :] = (
        cal["gain"][0 : cal["n_pol"], :, :]
        / cal_bandpass["gain"][0 : cal["n_pol"], :, :]
    )
    return cal_bandpass, cal_remainder


def vis_cal_bandpass(
    obs: dict, cal: dict, pyfhd_config: dict, logger: Logger
) -> tuple[dict, dict]:
    """
    Reduce the degrees of freedom on the per-frequency calibration amplitudes by averaging solutions
    together. Options include averaging over tiles which use a particular beamformer-to-receiver cable
    lengths/types, or averaging over all tiles for a global bandpass.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger for displaying errors and info to the log files

    Returns
    -------
    cal_bandpass : dict
        Calibration dictionary with bandpass gains
    cal_remainder : dict
        calibration dictionary with residuals after removing the bandpass gains
    """
    freq_use = np.nonzero(obs["baseline_info"]["freq_use"])[0]
    tile_use = np.nonzero(obs["baseline_info"]["tile_use"])[0]
    # Set a flag for global bandpass, will turn true if too many tiles are flagged
    global_bandpass = False

    # Initialize cal_bandpass and cal_remainder and transfer them in, if a file has been set (fits only supported right now)
    if pyfhd_config["cal_bp_transfer"] is not None:
        cal_bandpass, cal_remainder = transfer_bandpass(obs, cal, pyfhd_config, logger)
        if len(cal_bandpass.keys()) != 0 and len(cal_remainder.keys()) != 0:
            logger.info(
                f"Calibration Bandpass FITS file {pyfhd_config['cal_bp_transfer']} transferred in for cal_bandpass and cal_remainder"
            )
            return cal_bandpass, cal_remainder
    cal_bandpass = deepcopy(cal)
    cal_remainder = deepcopy(cal)

    cal_bandpass_gain = np.empty(cal["gain"].shape, dtype=np.complex128)
    cal_remainder_gain = np.empty(cal["gain"].shape, dtype=np.complex128)

    if pyfhd_config["cable_bandpass_fit"]:
        # Using preexisting file to extract information about which tiles have which cable length
        # cable_len = np.loadtxt(Path(pyfhd_config["input"], pyfhd_config["cable-reflection-coefficients"]), skiprows=1)[:, 2].flatten()
        cable_len_filepath = importlib_resources.files(
            "pyfhd.resources.instrument_config"
        ).joinpath(f"{pyfhd_config['instrument']}_cable_reflection_coefficients.txt")
        cable_len = np.loadtxt(cable_len_filepath, skiprows=1)[:, 2].flatten()

        # Taking tile information and cross-matching it with the nonflagged tiles array, resulting in nonflagged tile arrays grouped by cable length
        cable_length_ref = np.unique(cable_len)
        tile_use_arr = [0] * cable_length_ref.size
        for cable_i in range(cable_length_ref.size):
            tile_use_arr[cable_i] = np.where(
                (obs["baseline_info"]["tile_use"])
                & (cable_len == cable_length_ref[cable_i])
            )[0]
            if tile_use_arr[cable_i].size == 0:
                logger.warning(
                    "Too Many flagged tiles to implement bandpass cable averaging, using global bandpass."
                )
                global_bandpass = True
        # n_freq x 13 array. columns are frequency, 90m xx, 90m yy, 150m xx, 150m yy, 230m xx, 230m yy, 320m xx, 320m yy, 400m xx, 400m yy, 524m xx, 524m yy
        bandpass_arr = np.zeros(
            (obs["n_freq"], cal["n_pol"] * cable_length_ref.size + 1)
        )
        bandpass_arr[:, 0] = obs["baseline_info"]["freq"]
        bandpass_col_count = 1
        if pyfhd_config["auto_ratio_calibration"]:
            logger.info("auto_ratio_calibration is set, using global bandpass")
            global_bandpass = True
        for cable_i in range(cable_length_ref.size):
            # This is an option to calibrate over all tiles to find the 'global' bandpass. It will be looped over by the number
            # of cable lengths, and will redo the same calculation everytime. It is inefficient, but effective.
            if global_bandpass:
                tile_use_cable = tile_use
            else:
                tile_use_cable = tile_use_arr[cable_i]

            for pol_i in range(cal["n_pol"]):
                gain = cal["gain"][pol_i]
                # gain2 is a temporary variable used in place of the gain array for an added layer of safety
                if cable_i == 0 and pol_i == 0:
                    gain2 = np.zeros(cal["gain"].shape, dtype=np.complex128)
                # Only use gains from unflagged tiles and frequencies, and calculate the amplitude and phase
                gain_use = gain[freq_use, :][:, tile_use_cable]
                amp = np.abs(gain_use)
                # amp2 is a temporary variable used in place of the amp array for an added layer of safety
                amp2 = np.zeros((freq_use.size, tile_use_cable.size))
                # This is the normalization loop for each tile. If the mean of gain amplitudes over all frequencies is nonzero, then divide
                # the gain amplitudes by that number, otherwise make the gain amplitudes zero.
                for tile_i in range(tile_use_cable.size):
                    res_mean = resistant_mean(amp[:, tile_i], 2)
                    if res_mean != 0:
                        amp2[:, tile_i] = amp[:, tile_i] / res_mean
                    else:
                        amp2[:, tile_i] = 0
                # This finds the normalized gain amplitude mean per frequency over all tiles, which is the final bandpass per cable group.
                bandpass_single = np.empty(freq_use.size)
                # If this is slow, resistant_mean can be vectorized
                for f_i in range(freq_use.size):
                    bandpass_single[f_i] = resistant_mean(amp2[f_i, :], 2)
                # Want iterative to start at 1 (to not overwrite freq) and store final bandpass per cable group.
                bandpass_arr[freq_use, bandpass_col_count] = bandpass_single
                bandpass_col_count += 1
                # Fill temporary variable gain2, set equal to final bandpass per cable group for each tile that will use that bandpass.

                # Vectorize later if possible
                for tile_i in range(tile_use_cable.size):
                    gain2[pol_i, freq_use, tile_use_cable[tile_i]] = bandpass_single

                # For the last bit at the end of the cable
                if cable_i == cable_length_ref.size - 1:
                    # Set gain3 to the input gains
                    gain3 = cal["gain"][pol_i].copy()
                    # Set what will be passed back as the output gain as the final bandpass per cable type.
                    gain2_input = gain2[pol_i, :, :]
                    cal_bandpass_gain[pol_i] = gain2_input
                    # Set what will be passed back as the residual as the input gain divided by the final bandpass per cable type.
                    gain3[freq_use, :] /= gain2_input[freq_use, :]
                    cal_remainder_gain[pol_i] = gain3
        # Add Levine Memo bandpass to the gain solutions here if you wish
    else:
        bandpass_arr = np.zeros((obs["n_freq"], cal["n_pol"] + 1))
        bandpass_arr[:, 0] = obs["baseline_info"]["freq"]
        for pol_i in range(cal["n_pol"]):
            gain = cal["gain"][pol_i]
            gain_use = gain[freq_use, :][:, tile_use]
            amp = np.abs(gain_use)
            amp2 = np.zeros((freq_use.size, tile_use.size))
            for tile_i in range(tile_use.size):
                res_mean = resistant_mean(amp[:, tile_i], 2)
                if res_mean != 0:
                    amp2[:, tile_i] = amp[:, tile_i] / res_mean
                else:
                    amp2[:, tile_i] = 0

            bandpass_single = np.empty(freq_use.size)
            # If this is slow, resistant_mean can be vectorized
            for f_i in range(freq_use.size):
                bandpass_single[f_i] = resistant_mean(amp2[f_i, :], 2)

            bandpass_arr[freq_use, pol_i + 1] = bandpass_single
            # Work out the gain for the bandpass
            gain2 = np.zeros(gain.shape, dtype=np.complex128)
            gain3 = deepcopy(gain)
            # Vectorize later if possible
            for tile_i in range(obs["n_tile"]):
                gain2[freq_use, tile_i] = bandpass_single
                gain3[freq_use, tile_i] /= bandpass_single

            cal_bandpass_gain[pol_i, :, :] = gain2
            cal_remainder_gain[pol_i, :, :] = gain3

    cal_bandpass["gain"] = cal_bandpass_gain
    cal_remainder["gain"] = cal_remainder_gain
    return cal_bandpass, cal_remainder


def vis_cal_polyfit(
    obs: dict,
    cal: dict,
    auto_ratio: NDArray[np.float64] | None,
    pyfhd_config: dict,
    logger: Logger,
) -> dict:
    """
    Reduce the degrees of freedom on the per-frequency calibration amplitudes and phases by fitting
    the full frequency band with polynomials of a specified degree, with options for split polynomials
    over certain regions.

    In addition, fit cable reflections with a text file of fits, theoretical fits given cable length/type,
    or the finding the maximum in delay space. Any of these can then be used as an initial estimate in a
    hyperresolved FFT of the residual calibration solutions to fit the final mode, phase, and amplitude.
    Residual calibration solutions can optionally be made by using an incoherent mean, or a mean over
    all tiles which *do not* have the same cable length/type, to reduce bias in the residual, and
    the cable reflections can also be fit using just the phases to reduce dependency on the polyphase
    filter bank shape.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    auto_ratio: NDArray[np.float64] | None
        The estimation of the antenna-dependent bias through the square root of the autocorrelation
        visibilities normalized via a reference tile
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger for displaying errors and info to the log files

    Returns
    -------
    cal : dict
        Calibration dictionary with polynomial gain fits
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
        Is returned in case the digital_gain_jump_polyfit was set here.
    """
    # Keep the og_gain_arr for calculations later
    og_gain_arr = np.copy(cal["gain"])
    if (
        pyfhd_config["cal_reflection_mode_theory"]
        or pyfhd_config["cal_reflection_mode_file"]
        or pyfhd_config["cal_reflection_mode_delay"]
        or pyfhd_config["cal_reflection_hyperresolve"]
    ):
        if (
            pyfhd_config["cal_reflection_mode_theory"]
            and abs(pyfhd_config["cal_reflection_mode_theory"]) > 1
        ):
            cal_mode_fit = pyfhd_config["cal_reflection_mode_theory"]
        else:
            cal_mode_fit = True
    else:
        cal_mode_fit = False
    freq_use = np.where(obs["baseline_info"]["freq_use"])[0]
    tile_use = np.where(obs["baseline_info"]["tile_use"])[0]

    pre_dig_inds = np.where(obs["baseline_info"]["freq"][freq_use] < 187.515e6)
    if pre_dig_inds[0].size == 0:
        logger.warning(
            "No frequencies below 187.515MHz, using full band polyfit, digital gain jump polyfit disabled."
        )
        pyfhd_config["digital_gain_jump_polyfit"] = False
    else:
        f_d = np.max(pre_dig_inds[0])
        f_end = freq_use.size

    if ((f_d + 1) == f_end) and pyfhd_config["digital_gain_jump_polyfit"]:
        logger.warning(
            "No frequency found above 187.515MHz, using full band polyfit, digital gain jump polyfit disabled."
        )
        pyfhd_config["digital_gain_jump_polyfit"] = False

    # If the observation date is beyond the date where digital gain jump polyfit is required, disable it
    # Date is 2014-07-23 (23rd July 2014) or Julian Day 2456861.5000000
    if obs["jd0"] > 2456861.5000000 and pyfhd_config["digital_gain_jump_polyfit"]:
        logger.warning(
            "Observation date is beyond the date where digital gain jump polyfit is required, using full band polyfit, digital gain jump polyfit disabled."
        )
        pyfhd_config["digital_gain_jump_polyfit"] = False

    # If the amp_degree or phase_degree weren't used, then apply the defaults
    if not pyfhd_config["cal_amp_degree_fit"]:
        pyfhd_config["cal_amp_degree_fit"] = 2
    if not pyfhd_config["cal_phase_degree_fit"]:
        pyfhd_config["cal_phase_degree_fit"] = 1

    # If you wish to find any steps that are outliers beyond 5sigma, and remove them add that code here.
    # The cal_step_fit option isn't in the eor_defaults_wrapper or defined in the FHD dictionary.

    # Polynomial fitting over the frequency band
    gain_residual = np.empty((cal["n_pol"], obs["n_freq"], obs["n_tile"]))
    # create amp_params with the shape of (n_pol, n_tile, amp_degree + 1)
    # If we're using the digital_gain_jump_polyfit add extra dimension to amp_params
    if pyfhd_config["digital_gain_jump_polyfit"]:
        cal["amp_params"] = np.empty(
            (
                cal["n_pol"],
                obs["n_tile"],
                pyfhd_config["cal_amp_degree_fit"],
                pyfhd_config["cal_amp_degree_fit"],
            )
        )
    else:
        cal["amp_params"] = np.empty(
            (cal["n_pol"], obs["n_tile"], pyfhd_config["cal_amp_degree_fit"] + 1)
        )
    cal["phase_params"] = np.empty(
        (cal["n_pol"], obs["n_tile"], pyfhd_config["cal_phase_degree_fit"] + 1)
    )
    for pol_i in range(cal["n_pol"]):
        gain_arr = np.copy(cal["gain"][pol_i])
        gain_amp = np.abs(gain_arr)
        gain_phase = np.arctan2(gain_arr.imag, gain_arr.real)
        for tile_i in range(obs["n_tile"]):
            gain = np.squeeze(gain_amp[freq_use, tile_i])
            gain_fit = np.zeros(obs["n_freq"])
            # Pre and post digital gain jump separately for highband MWA data
            if pyfhd_config["digital_gain_jump_polyfit"]:
                if pre_dig_inds[0].size > 0:
                    fit_params1 = (
                        np.polynomial.Polynomial.fit(
                            freq_use[0 : f_d + 1],
                            gain[0 : f_d + 1],
                            deg=pyfhd_config["cal_amp_degree_fit"] - 1,
                        )
                        .convert()
                        .coef
                    )
                    fit_params2 = (
                        np.polynomial.Polynomial.fit(
                            freq_use[f_d + 1 : f_end],
                            gain[f_d + 1 : f_end],
                            deg=pyfhd_config["cal_amp_degree_fit"] - 1,
                        )
                        .convert()
                        .coef
                    )
                    for di in range(pyfhd_config["cal_amp_degree_fit"]):
                        gain_fit[freq_use[0] : freq_use[f_d] + 1] += fit_params1[di] * (
                            # Had to use the size as use freq_use[f_d] could cause
                            # off by one indexing errors (due to freq_use[0] potentially being
                            # 1 or 0 which throws off the np.arange indexing)
                            np.arange(gain_fit[freq_use[0] : freq_use[f_d] + 1].size)
                            ** di
                        )
                        gain_fit[freq_use[f_d + 1] : freq_use[f_end - 1] + 1] += (
                            fit_params2[di]
                            * (
                                np.arange(freq_use[f_end - 1] - freq_use[f_d + 1] + 1)
                                + freq_use[f_d + 1]
                            )
                            ** di
                        )
                    # Do notice this is saving the coefficients on a per row basis, so fit_params1 a,b will be [pol_i, tile_i, 0, :]
                    # While fit_params2 a, b coefficients will be in [pol_i, tile_i, 1, :]
                    fit_params = np.vstack([fit_params1, fit_params2])
                    cal["amp_params"][pol_i, tile_i] = fit_params
                else:
                    logger.warning(
                        "digital_gain_jump_polyfit only works with highband mwa data. Full band polyfit applied instead."
                    )
            # Fit for amplitude
            else:
                fit_params = (
                    np.polynomial.Polynomial.fit(
                        freq_use, gain, deg=pyfhd_config["cal_amp_degree_fit"]
                    )
                    .convert()
                    .coef
                )
                cal["amp_params"][pol_i, tile_i, :] = fit_params
                for di in range(pyfhd_config["cal_amp_degree_fit"]):
                    gain_fit += fit_params[di] * np.arange(obs["n_freq"]) ** di

            gain_residual[pol_i, :, tile_i] = gain_amp[:, tile_i] - gain_fit

            # Fit for phase
            phase_use = np.unwrap(np.squeeze(gain_phase[freq_use, tile_i]))
            phase_params = (
                np.polynomial.Polynomial.fit(
                    freq_use, phase_use, pyfhd_config["cal_phase_degree_fit"]
                )
                .convert()
                .coef
            )
            cal["phase_params"][pol_i, tile_i, :] = phase_params
            phase_fit = np.zeros(obs["n_freq"])
            for di in range(phase_params.size):
                phase_fit += phase_params[di] * np.arange(obs["n_freq"]) ** di
            gain_arr[:, tile_i] = gain_fit * np.exp(1j * phase_fit)
        cal["gain"][pol_i] = gain_arr
    # Cable Reflection Fitting
    if cal_mode_fit:
        if pyfhd_config["cal_reflection_mode_file"]:
            logger.info(
                "Using mwa calibration reflections fits from instrument_config/mwa_cable_reflection_coefficients.txt."
            )
            cable_len_filepath = importlib_resources.files(
                "pyfhd.resources.instrument_config"
            ).joinpath(
                f"{pyfhd_config['instrument']}_cable_reflection_coefficients.txt"
            )
            cable_reflections = np.loadtxt(cable_len_filepath, skiprows=1).transpose()
            cable_length = cable_reflections[2]
            tile_ref_flag = np.minimum(
                np.maximum(np.zeros_like(cable_reflections[4]), cable_reflections[4]),
                np.ones_like(cable_reflections[4]),
            )
            tile_mode_X = cable_reflections[5]
            tile_amp_X = cable_reflections[6]
            tile_phase_X = cable_reflections[7]
            tile_mode_Y = cable_reflections[8]
            tile_amp_Y = cable_reflections[9]
            tile_phase_Y = cable_reflections[10]

            # Modes in fourier transform units
            mode_i_arr = np.zeros((cal["n_pol"], obs["n_tile"]))
            mode_i_arr[0, :] = tile_mode_X * tile_ref_flag
            mode_i_arr[1, :] = tile_mode_Y * tile_ref_flag

            amp_arr = np.vstack([tile_amp_X, tile_amp_Y])
            phase_arr = np.vstack[[tile_phase_X, tile_phase_Y]]

        elif pyfhd_config["cal_reflection_mode_theory"]:
            logger.info(
                "Using theory calculation in nominal reflection mode calibration."
            )
            # Get the nominal tile lengths and velocity factors
            cable_len_filepath = importlib_resources.files(
                "pyfhd.resources.instrument_config"
            ).joinpath(f"{pyfhd_config['instrument']}_cable_length.txt")
            cable_length_data = np.loadtxt(cable_len_filepath, skiprows=1).transpose()
            cable_length = cable_length_data[2]
            cable_vf = cable_length_data[3]
            tile_ref_flag = np.minimum(np.maximum(0, cable_length_data[4]), 1)

            # Nominal Reflect Time
            reflect_time = (2 * cable_length) / (c.value * cable_vf)
            bandwidth = (
                (
                    np.max(obs["baseline_info"]["freq"])
                    - np.min(obs["baseline_info"]["freq"])
                )
                * obs["n_freq"]
            ) / (obs["n_freq"] - 1)
            # Modes in fourier transform units
            mode_i_arr = np.tile(
                bandwidth * reflect_time * tile_ref_flag, [cal["n_pol"], 1]
            )

        elif pyfhd_config["cal_reflection_mode_delay"]:
            logger.info(
                "Using calibration delay spectrum to calculate nominal reflection modes."
            )
            spec_mask = np.zeros(obs["n_freq"])
            spec_mask[freq_use] = 1
            freq_cut = np.where(spec_mask == 0)
            # IDL uses forward FFT by default
            spec_psf = np.abs(np.fft.fftn(spec_mask, norm="forward"))
            spec_inds = np.arange(obs["n_freq"] // 2)
            spec_psf = spec_psf[spec_inds]
            mode_test = np.zeros(obs["n_freq"] // 2)
            for pol_i in range(cal["n_pol"]):
                for ti in range(tile_use.size):
                    tile_i = tile_use[ti]
                    spec0 = np.abs(np.fft.fftn(gain_residual[pol_i, tile_i]))
                    mode_test += spec0[spec_inds]
            psf_mask = np.zeros(obs["n_freq"] // 2)

            if freq_cut[0].size > 0:
                psf_mask[np.where(spec_psf > (np.max(spec_psf) / 1000))] = 1
                # Replaces IDL smooth with edge_truncate
                psf_mask = uniform_filter(psf_mask, size=5, mode="nearest")
                mask_i = np.nonzero(psf_mask)
                if mask_i[0].size > 0:
                    mode_test[mask_i] = 0
            mode_i_arr = np.zeros((cal["n_pol"], obs["n_tile"])) + np.argmax(mode_test)

        # Fit only certain cable lengths
        # Positive length indicates fit mode, negative length indicates exclude mode
        # This is currently assuming cal_mode_fit is an integer or number, not an array!
        # If you need an array to fit or exclude cable lengths, then create another option for it
        # in the config and adjust the code accordingly. Ensure every config option only has one purpose.
        if auto_ratio is None and cal_mode_fit != 1:
            tile_ref_logic = np.zeros(obs["n_tile"])
            if cal_mode_fit > 0:
                cable_cut_i = np.where(cable_length != cal_mode_fit)
                if cable_cut_i[0].size > 0:
                    tile_ref_logic[cable_cut_i] = 1
            else:
                cable_cut_i = np.where(cable_length == abs(cal_mode_fit))
                if cable_cut_i[0].size > 0:
                    tile_ref_logic[cable_cut_i] = 1
            cable_cut = np.nonzero(tile_ref_logic)
            tile_ref_flag = np.ones(obs["n_tile"])
            if cable_cut[0].size > 0:
                tile_ref_flag[cable_cut] = 0
            mode_i_arr *= tile_ref_flag

        cal["mode_params"] = np.empty([cal["n_pol"], obs["n_tile"], 3])
        for pol_i in range(cal["n_pol"]):
            # Divide the polyfit to reveal the residual cable reflections better
            gain_arr = og_gain_arr[pol_i] / cal["gain"][pol_i]
            for ti in range(tile_use.size):
                tile_i = tile_use[ti]
                mode_i = mode_i_arr[pol_i, tile_i]
                if mode_i == 0:
                    continue
                else:
                    # Options to hyperresolve or fit the reflection modes/amp/phase given the nominal calculations
                    if pyfhd_config["cal_reflection_hyperresolve"]:
                        # start with nominal cable length
                        mode0 = mode_i
                        # overresolve the FT used for the fit (normal resolution would be dmode=1)
                        dmode = 0.05
                        # range around the central mode to test
                        nmodes = 101
                        # array of modes to try
                        modes = (np.arange(nmodes) - nmodes // 2) * dmode + mode0
                        # reshape for ease of computing
                        modes = rebin(modes, (freq_use.size, nmodes)).T

                        if auto_ratio is not None:
                            # Find tiles which will *not* be accidently coherent in their cable reflection in order to reduce bias
                            inds = np.where(
                                (obs["baseline_info"]["tile_use"])
                                & (mode_i_arr[pol_i, :] > 0)
                                & ((np.abs(mode_i_arr[pol_i, :] - mode_i)) > 0.01)
                            )
                            # mean over frequency for each tile
                            freq_mean = np.nanmean(auto_ratio[pol_i], axis=0)
                            norm_autos = auto_ratio[pol_i] / rebin(
                                freq_mean, (obs["n_freq"], obs["n_tile"])
                            )
                            # mean over all tiles which *are not* accidently coherent as a func of freq
                            incoherent_mean = np.nanmean(norm_autos[:, inds[0]], axis=1)
                            # Residual and normalized (using incoherent mean) auto-correlation
                            resautos = (
                                norm_autos[:, tile_i] / incoherent_mean
                            ) - np.nanmean(norm_autos[:, tile_i] / incoherent_mean)
                            gain_temp = rebin(
                                resautos[freq_use], (nmodes, freq_use.size)
                            )
                        else:
                            # dimension manipulation, add dim for mode fitting
                            # Subtract the mean so aliasing is reduced in the dft cable fitting
                            gain_temp = rebin(
                                gain_arr[freq_use, tile_i]
                                - np.mean(gain_arr[freq_use, tile_i]),
                                (nmodes, freq_use.size),
                            )
                        # freq_use matrix to multiply/collapse in fit
                        freq_mat = rebin(freq_use, (nmodes, freq_use.size))
                        # Perform DFT of gains to test modes
                        test_fits = np.sum(
                            np.exp(1j * 2 * np.pi / obs["n_freq"] * modes * freq_mat)
                            * gain_temp,
                            axis=1,
                        )
                        # Pick out highest amplitude fit (mode_ind gives the index of the mode)
                        amp_use = np.max(np.abs(test_fits)) / freq_use.size
                        mode_ind = np.argmax(np.abs(test_fits))
                        # Phase of said fit
                        phase_use = np.arctan2(
                            test_fits[mode_ind].imag, test_fits[mode_ind].real
                        )
                        mode_i = modes[mode_ind, 0]

                        # Using the mode selected from the gains, optionally use the phase to find the amp and phase
                        if auto_ratio is not None:
                            # Find tiles which will not be accidently coherent in their cable reflection in order to reduce bias
                            inds = np.where(
                                (obs["baseline_info"]["tile_use"])
                                & (mode_i_arr[pol_i, :] > 0)
                                & (np.abs(mode_i_arr[pol_i, :] - mode_i) > 0.01)
                            )
                            residual_phase = np.arctan2(
                                gain_arr[freq_use, :].imag, gain_arr[freq_use, :].real
                            )
                            incoherent_residual_phase = residual_phase[
                                :, tile_i
                            ] - np.nanmean(residual_phase[:, inds[0]], axis=1)
                            test_fits = np.sum(
                                np.exp(
                                    1j * 2 * np.pi / obs["n_freq"] * mode_i * freq_use
                                )
                                * incoherent_residual_phase
                            )
                            # Factor of 2 from fitting just the phase
                            amp_use = 2 * np.abs(test_fits) / freq_use.size
                            # Factor of pi/2 from just fitting the phase
                            phase_use = (
                                np.arctan2(test_fits.imag, test_fits.real) + np.pi / 2
                            )
                    elif pyfhd_config["cal_reflection_mode_file"]:
                        # Use predetermined fits
                        amp_use = amp_arr[pol_i, tile_i]
                        phase_use = phase_arr[pol_i, tile_i]
                    else:
                        # Use nominal delay mode, but fit amplitude and phase of reflections
                        mode_fit = np.sum(
                            np.exp(1j * 2 * np.pi / obs["n_freq"] * mode_i * freq_use)
                            * gain_arr[freq_use, tile_i]
                        )
                        amp_use = np.abs(mode_fit) / freq_use[0].size
                        phase_use = np.arctan2(mode_fit.imag, mode_fit.real)

                    gain_mode_fit = amp_use * np.exp(
                        -1j
                        * 2
                        * np.pi
                        * (mode_i * np.arange(obs["n_freq"]) / obs["n_freq"])
                        + 1j * phase_use
                    )
                    if auto_ratio is not None:
                        # Only fit for the cable reflection in the phases
                        cal["gain"][pol_i, :, tile_i] *= np.exp(1j * gain_mode_fit.imag)
                    else:
                        cal["gain"][pol_i, :, tile_i] *= 1 + gain_mode_fit
                    cal["mode_params"][pol_i, tile_i] = np.array(
                        [mode_i, amp_use, phase_use]
                    )
    return cal, pyfhd_config


def vis_cal_auto_fit(
    obs: dict,
    cal: dict,
    vis_auto: NDArray[np.float64],
    vis_auto_model: NDArray[np.float64],
    auto_tile_i: NDArray[np.integer],
) -> dict:
    """
    Solve for each tile's calibration amplitude via the square root of the ratio of the data autocorrelation
    to the model autocorrelation using the definition of a gain. Then, remove dependence on the correlated
    noise term in the autocorrelations by scaling this amplitude down to the crosscorrelations gain via a
    simple, linear fit. Build a full, scaled autocorrelation gain solution by adding in the phase term via
    the crosscorrelation gains.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    vis_auto : NDArray[np.float64]
        Data autocorrelations
    vis_auto_model : NDArray[np.float64]
        Simulated model autocorrelations
    auto_tile_i : NDArray[np.integer]
        Index array of the tile array that have defined autocorrelations

    Returns
    -------
    cal: dict
        Calibration dictionary with scaled autocorrelation gains
    """
    freq_i_use = np.nonzero(obs["baseline_info"]["freq_use"])
    freq_i_flag = np.where(obs["baseline_info"]["freq_use"] == 0)[0]
    # If the number of frequencies not being used is above 0, then ignore the frequencies surrounding them.
    if freq_i_flag.size > 0:
        freq_flag = np.zeros(obs["n_freq"])
        freq_flag[freq_i_use] = 1
        for freq_i in range(freq_i_flag.size):
            minimum = max(0, freq_i_flag[freq_i] - 1)
            maximum = min(obs["n_freq"], freq_i_flag[freq_i] + 2)
            freq_flag[minimum:maximum] = 0
        freq_i_use = np.nonzero(freq_flag)
    # Vectorized loop for via_cal_auto_fit lines 45-55 in IDL
    # However the logic still indexes the full 128 tiles, so need to shove
    # outputs into an empty array of correct size
    # We're not using the cross polarizations if they are present
    auto_gain = np.empty((cal["n_pol"], obs["n_freq"], obs["n_tile"]))
    auto_gain[:, :, auto_tile_i] = np.sqrt(
        vis_auto[: cal["n_pol"]] * weight_invert(vis_auto_model[: cal["n_pol"]])
    )
    gain_cross = cal["gain"]
    fit_slope = np.empty((cal["n_pol"], obs["n_tile"]))
    fit_offset = np.empty_like(fit_slope)

    # Didn't vectorize as the polyfit won't be vectorized
    for pol_i in range(cal["n_pol"]):
        for tile in range(auto_tile_i.size):
            tile_idx = auto_tile_i[tile]
            phase_cross_single = np.arctan2(
                gain_cross[pol_i, :, tile_idx].imag, gain_cross[pol_i, :, tile_idx].real
            )

            gain_auto_single = np.abs(auto_gain[pol_i, :, tile_idx])
            gain_cross_single = np.abs(gain_cross[pol_i, :, tile_idx])

            # mask out any NaN values; numpy doesn't like them,
            # I assume the IDL equiv function just masks them?
            # or maybe we need to do a catch for NaNs here, and abandon all
            # hope for a fit if there are NaNs?
            notnan = np.nonzero(
                (~np.isnan(gain_auto_single[freq_i_use]))
                & (~np.isnan(gain_cross_single[freq_i_use]))
            )
            gain_auto_single_fit = gain_auto_single[freq_i_use][notnan]
            gain_cross_single_fit = gain_cross_single[freq_i_use][notnan]

            # linfit from IDL uses chi-square error calculations to do the linear fit, instead of least squares.
            # The polynomial fit uses least square method
            x = np.vstack([gain_auto_single_fit, np.ones(gain_auto_single_fit.size)]).T
            fit_single = np.linalg.lstsq(x, gain_cross_single_fit, rcond=None)[0]
            # IDL gives the solution in terms of [A, B] while Python does [B, A] assuming we're
            # solving the equation y = A + Bx
            cal["gain"][pol_i, :, tile_idx] = (
                gain_auto_single * fit_single[0] + fit_single[1]
            ) * np.exp(1j * phase_cross_single)
            fit_slope[pol_i, tile_idx] = fit_single[0]
            fit_offset[pol_i, tile_idx] = fit_single[1]
    cal["auto_scale"] = np.sum(fit_slope, axis=1) / auto_tile_i.size
    cal["auto_params"] = np.empty([cal["n_pol"], cal["n_pol"], obs["n_tile"]])
    cal["auto_params"][0, :, :] = fit_offset
    cal["auto_params"][1, :, :] = fit_slope
    return cal


def vis_calibration_apply(
    vis_arr: NDArray[np.complex128],
    obs: dict,
    cal: dict,
    vis_model_arr: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    logger: Logger,
) -> tuple[NDArray[np.complex128], dict]:
    """
    Apply the calibration solutions to the input, data visibilities to create calibrated, data visibilities using
    the definition of the gains.

    Definition of the gain:
    (visibility for baseline of tile i and tile j) = (gain of tile i) (gain of tile j) (model visibility for baseline of tile i and tile j)

    If only two othogonal polarizations were used to calibrate, calculate the phase offset between the two orthogonal dipoles to
    solve for a degeneracy in the cross polarizations. The formula can be derived by multiplying the gains by an equal and opposite
    phase in the linear least square solver and solving for the phase when the partial derivative w.r.t the offset is 0.

    Parameters
    ----------
    vis_arr : NDArray[np.complex128]
        Uncalibrated data visiblities
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    vis_model_arr : NDArray[np.complex128]
        Simulated model visibilites
    vis_weights : NDArray[np.float64]
        Weights (flags) of the visibilities
    logger : Logger
        pyfhd's logger for displaying errors and info to the log files

    Returns
    -------
    vis_arr : NDArray[np.complex128]
        Calibrated visibilities
    cal : dict
        Calibration dictionary
    """
    # tile numbering starts at 1
    tile_a_i = obs["baseline_info"]["tile_a"] - 1
    tile_b_i = obs["baseline_info"]["tile_b"] - 1

    n_pol_vis = vis_arr.shape[0]
    gain_pol_arr1 = [0, 1, 0, 1]
    gain_pol_arr2 = [0, 1, 1, 0]

    # OK, it really makes sense to use native python functionality here
    # We're just trying to match up the frequency-dependent gains to the
    # correct baselines, and apply them. Can use `meshgrid` here instead of
    # `rebin`, which will make 2D indexing arrays, so we can directly leave
    # the gain arrays in the correct shape and index the directly. Using `rebin`
    # means we have to flatten them
    inds_a_baseline, inds_a_freqs = np.meshgrid(tile_a_i, np.arange(obs["n_freq"]))
    inds_b_baseline, inds_b_freqs = np.meshgrid(tile_b_i, np.arange(obs["n_freq"]))

    for pol_i in range(n_pol_vis):
        gain_arr1 = cal["gain"][gain_pol_arr1[pol_i], :, :]
        gain_arr2 = cal["gain"][gain_pol_arr2[pol_i], :, :]

        vis_gain = gain_arr1[inds_a_freqs, inds_a_baseline] * np.conjugate(
            gain_arr2[inds_b_freqs, inds_b_baseline]
        )

        vis_arr[pol_i, :, :] *= weight_invert(vis_gain, use_abs=False)

    # We haven't run FHD in a way that uses 4 pols yet so this is all
    # untested
    if n_pol_vis == 4:
        if isinstance(vis_model_arr, np.ndarray) and isinstance(
            vis_weights, np.ndarray
        ):
            # This if statement replaces vis_calibrate_crosspol_phase
            # as this was the only place where the function was used
            # Note inside vis_calibrate_crosspol_phase there is a
            # if n_pol_vis == 4 check hence only run if n_pol_vis == 4
            # This code should calculate the phase fit between the X and Y
            # antenna polarizations.
            # Use the xx flags (yy should be identitical at this point)

            # this is num baselines per time step
            n_baselines = int(len(tile_a_i) / obs["n_times"])
            # reshape from (n_freq, n_baselines*n_times) to (n_freq, n_times, n_baselines). Turns out due to the row major vs col major difference
            # between IDL and python, this shape also changes
            new_shape = (obs["n_freq"], obs["n_times"], n_baselines)
            weights_use = np.reshape(vis_weights[0, :, :], new_shape)
            # carried over from FHD code
            weights_use = np.maximum(weights_use, np.zeros_like(weights_use))
            weights_use = np.minimum(weights_use, np.ones_like(weights_use))

            # Average the visbilities in time
            axis_avg = 1
            vis_xy = np.reshape(vis_arr[2, :, :], new_shape)
            vis_xy = np.sum(vis_xy * weights_use, axis=axis_avg)
            vis_yx = np.reshape(vis_arr[3, :, :], new_shape)
            vis_yx = np.sum(vis_yx * weights_use, axis=axis_avg)

            model_xy = np.reshape(vis_model_arr[2, :, :], new_shape)
            model_xy = np.sum(model_xy * weights_use, axis=axis_avg)
            model_yx = np.reshape(vis_model_arr[3, :, :], new_shape)
            model_yx = np.sum(model_yx * weights_use, axis=axis_avg)

            # Remove Zeros
            weight = np.sum(weights_use, axis=axis_avg)
            i_use = np.nonzero(weight)
            vis_xy = np.squeeze(vis_xy[i_use])
            vis_yx = np.squeeze(vis_yx[i_use])
            model_xy = np.squeeze(model_xy[i_use])
            model_yx = np.squeeze(model_yx[i_use])

            vis_sum = np.sum(np.conjugate(vis_xy) * model_xy) + np.sum(
                vis_yx * np.conjugate(model_yx)
            )
            cross_phase = np.arctan2(vis_sum.imag, vis_sum.real)

            logger.info(
                f"Phase fit between X and Y antenna polarizations: {cross_phase}"
            )

            cal["cross_phase"] = cross_phase

        vis_arr[2, :, :] *= np.exp(1j * 0)
        vis_arr[3, :, :] *= np.exp(-1j * 0)

    return vis_arr, cal


def vis_baseline_hist(
    obs: dict,
    params: dict,
    vis_cal: NDArray[np.complex128],
    vis_model_arr: NDArray[np.complex128],
) -> dict:
    """
    Create diagnostic histograms of both the mean and sigma for the percent difference between calibrated
    data and simulated model visibilities as a function of baseline length.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    params : dict
        Visibility metadata dictionary
    vis_cal : NDArray[np.complex128]
        Calibrated data visibilities
    vis_model_arr : NDArray[np.complex128]
        Simulated model visibilites

    Returns
    -------
    vis_baseline_hist : dict
        Dictionary of the mean and sigma histograms for the percent difference between calibrated data
        and simulated model visibilities as a function of baseline length.
    """
    kx_arr = params["uu"] / obs["kpix"]
    ky_arr = params["vv"] / obs["kpix"]
    kr_arr = np.sqrt(kx_arr**2 + ky_arr**2)
    # take the transpose of this, given our `vis_cal` and `vis_mode_arr` are the
    # transpose of the original FHD code
    dist_arr = np.outer(kr_arr, obs["baseline_info"]["freq"]).transpose() * obs["kpix"]
    dist_hist, bins, dist_ri = histogram(
        dist_arr, min=obs["min_baseline"], max=obs["max_baseline"], bin_size=5.0
    )

    vis_res_ratio_mean = np.zeros([obs["n_pol"], bins.size])
    vis_res_sigma = np.zeros([obs["n_pol"], bins.size])

    for pol_i in range(obs["n_pol"]):
        for bin_i in range(bins.size):
            if dist_hist[bin_i] > 0:
                inds = dist_ri[dist_ri[bin_i] : dist_ri[bin_i + 1]]
                model_vals = (vis_model_arr[pol_i]).flatten()[inds]
                wh_noflag = np.where(np.abs(model_vals) > 0)[0]
                if wh_noflag.size > 0:
                    inds = inds[wh_noflag]
                    # inds changed so we need to update model_vals
                    # otherwise it has a different shape than vis_cal_use below
                    # causing errors
                    model_vals = (vis_model_arr[pol_i]).flatten()[inds]
                else:
                    continue
                # if Keyword_Set(calibration_visibilities_subtract) THEN BEGIN
                # but calibration_visibilities_subtract isn't a function keyword
                # so we'll only translate the False of that statement
                # vis_cal_use = (vis_cal[pol_i].transpose()).flatten()[inds]
                vis_cal_use = (vis_cal[pol_i]).flatten()[inds]
                vis_res_ratio_mean[pol_i, bin_i] = np.mean(
                    np.abs(vis_cal_use - model_vals)
                ) / np.mean(np.abs(model_vals))
                vis_res_sigma[pol_i, bin_i] = np.sqrt(
                    np.var(np.abs(vis_cal_use - model_vals))
                ) / np.mean(np.abs(model_vals))

            else:
                continue
    return {
        "baseline_length": bins.size,
        "vis_res_ratio_mean": vis_res_ratio_mean,
        "vis_res_sigma": vis_res_sigma,
    }


def cal_auto_ratio_divide(
    obs: dict,
    cal: dict,
    vis_auto: NDArray[np.float64],
    auto_tile_i: NDArray[np.integer],
) -> tuple[dict, NDArray[np.float64]]:
    """
    Remove antenna-dependent parameters (i.e. cable reflections) from the calculated gains
    to reduce the bias on individual tile variation before the creation of averaged quantities
    like the global bandpass. Antenna-dependent parameters are estimated from the square root of the
    autocorrelation visibilities normalized via a reference tile.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    vis_auto : NDArray[np.float64]
        Data autocorrelations
    auto_tile_i : NDArray[np.integer]
        Index array of the tile array that have defined autocorrelations

    Returns
    -------
    cal : dict
        Calibration dictionary with gains that have reduced antenna-dependent bias
    auto_ratio : NDArray[np.float64]
        The estimation of the antenna-dependent bias through the square root of the autocorrelation
        visibilities normalized via a reference tile.
    """

    auto_ratio = np.empty([cal["n_pol"], obs["n_freq"], obs["n_tile"]])
    # This should be possible to Vectorize if it's slow
    for pol_i in range(cal["n_pol"]):
        # fhd_struct_init_cal puts the ref_antenna as 1 if it's not set, which is never appears to be
        v0 = vis_auto[pol_i, :, auto_tile_i[cal["ref_antenna"]]]
        auto_ratio[pol_i, :, auto_tile_i] = np.sqrt(
            vis_auto[pol_i, :, np.arange(auto_tile_i.size)] * weight_invert(v0)
        )
        cal["gain"][pol_i] = cal["gain"][pol_i] * weight_invert(auto_ratio[pol_i])
    return cal, auto_ratio


def cal_auto_ratio_remultiply(
    cal: dict, auto_ratio: NDArray[np.float64], auto_tile_i: NDArray[np.integer]
) -> dict:
    """
    Return antenna-dependent parameters to the calculated gains after averaged quantities
    have been calculated. The antenna-dependent parameters are captured via the square root of
    the referenced autocorrelation visiblities calculated in cal_auto_ratio_divide.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    auto_ratio : NDArray[np.float64]
        Square root of the autocorrelation visibilities normalized via a reference tile
    auto_tile_i : NDArray[np.integer]
        Index array of the tile array that have defined autocorrelations

    Returns
    -------
    cal: dict
        Calibration dictonary containing the reformed gain
    """
    # Replaced for loop in remultiply, this should remultiply by the auto_ratios
    cal["gain"][: cal["n_pol"], :, auto_tile_i] = cal["gain"][
        : cal["n_pol"], :, auto_tile_i
    ] * np.abs(auto_ratio[: cal["n_pol"], :, auto_tile_i])
    return cal


def calculate_adaptive_gain(
    gain_list: NDArray[np.float64],
    convergence_list: NDArray[np.float64],
    iter: int,
    base_gain: int | float,
    final_convergence_estimate: float | None = None,
):
    """
    Perform a Kalman filter to calculate the best weighting to use in the next iteration of the
    linear least squares fitting between the data and simulated model, which reduces the number
    of total iterations till convergence.

    The Kalman filter takes previous convergence values, or the percent change in calibration solution from
    the previous iteration, to calculate the relative weight to give to the next calculated iteration versus
    the previous iteration. For example, a gain (relative weight in this context) of 1 would give the old
    solution and the new solution the same weighting, and thus the next guess in the linear least square solver
    is the average of the new and old solution. A gain (relative weight) of 2 would give the new solution twice
    as much weight at the old solution, and thus the next guess in the linear least square solver is the sum of
    the old solution and twice the new solution, all divided by three.

    Parameters
    ----------
    gain_list : NDArray[np.float64]
        Relative weighting between the previous iteration and new iteration in the linear least squares
        solver for calibration solutions
    convergence_list : NDArray[np.float64]
        An array of the percent change in the calibration solutions between one iteration and the next
    iter : int
        The current iteration in the linear least squares solver
    base_gain : int|float
        The initial relative weighting between the previous iteration and new iteration in the linear
        least squares solver for calibration solutions
    final_convergence_estimate : float, optional
        The prediction of the percent change of the previous iteration and the forward model estimate from
        the Kalman filter, by default None

    Returns
    -------
    gain : float
        Relative weighting between the previous iteration and the new iteration in the calibration
        linear least squares solver
    """
    if iter > 2:
        # To calculate the best gain to use, compare the past gains that have been used
        # with the resulting convergences to estimate the best gain to use.
        # Algorithmically, this is a Kalman filter.
        # If forward modeling proceeds perfectly, the convergence metric should
        # asymptotically approach a final value.
        # We can estimate that value from the measured changes in convergence
        # weighted by the gains used in each previous iteration.
        # For some applications such as calibration this may be known in advance.
        # In calibration, it is expressed as the change in a
        # value, in which case the final value should be zero.
        if final_convergence_estimate is None:
            est_final_conv = np.zeros(iter - 1)
            for i in range(iter - 1):
                final_convergence_test = (
                    (1 + gain_list[i]) * convergence_list[i + 1] - convergence_list[i]
                ) / gain_list[i]
                # The convergence metric is strictly positive, so if the estimated final convergence is
                # less than zero, force it to zero.
                est_final_conv[i] = np.max((0, final_convergence_test))
            # Because the estimate may slowly change over time, only use the most recent measurements.
            final_convergence_estimate = np.median(est_final_conv[max(iter - 5, 0) :])
        last_gain = gain_list[iter - 1]
        last_conv = convergence_list[iter - 2]
        new_conv = convergence_list[iter - 1]
        # The predicted convergence is the value we would get if the new model calculated
        # in the previous iteration was perfect. Recall that the updated model that is
        # actually used is the gain-weighted average of the new and old model,
        # so the convergence would be similarly weighted.
        predicted_conv = (final_convergence_estimate * last_gain + last_conv) / (
            base_gain + last_gain
        )
        # If the measured and predicted convergence are very close, that indicates
        # that our forward model is accurate and we can use a more aggressive gain
        # If the measured convergence is significantly worse (or better!) than predicted,
        # that indicates that the model is not converging as expected and
        # we should use a more conservative gain.
        delta = (predicted_conv - new_conv) / (
            (last_conv - final_convergence_estimate) / (base_gain + last_gain)
        )
        new_gain = 1 - abs(delta)
        # Average the gains to prevent oscillating solutions.
        new_gain = (new_gain + last_gain) / 2
        # For some reason base_gain can be a numpy float in testing so putting in a tuple solves this.
        gain = np.max((base_gain / 2, new_gain))

    else:
        gain = base_gain
    gain_list[iter] = gain

    return gain

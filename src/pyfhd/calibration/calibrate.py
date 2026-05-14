import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
from logging import Logger
from pyfhd.calibration.calibration_utils import (
    vis_extract_autocorr,
    vis_cal_auto_init,
    vis_calibration_flag,
    vis_cal_bandpass,
    vis_cal_polyfit,
    vis_cal_auto_fit,
    vis_calibration_apply,
    vis_baseline_hist,
    cal_auto_ratio_divide,
    cal_auto_ratio_remultiply,
)
from pyfhd.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from pyfhd.pyfhd_tools.pyfhd_utils import resistant_mean, reshape_and_average_in_time
from pyfhd.plotting.calibration import plot_cals


def calibrate(
    obs: dict,
    params: dict,
    vis_arr: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    vis_model_arr: NDArray[np.complex128],
    pyfhd_config: dict,
    logger: Logger,
) -> tuple[NDArray[np.complex128], dict, dict]:
    """
    Solve for the amplitude and phase of the electronic response of each tile or station, and apply these
    calibration solutions to the raw, data visiblities. Various options for initial estimates, time/tile averaging,
    and polynomial/cable reflections fitting are available.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    params : dict
        Visibility metadata dictionary
    vis_arr : NDArray[np.complex128]
        Uncalibrated data visiblities
    vis_weights : NDArray[np.float64]
        Weights (flags) of the visibilities
    vis_model_arr : NDArray[np.complex128]
        Simulated model visibilites
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger for displaying errors and info to the log files

    Returns
    -------
    vis_cal : tuple[NDArray[np.complex128]
        The calibrated data visibilities
    cal : dict
        The updated calibration dictionary
    obs : dict
        Updated observation metadata dictionary
    pyfhd_config : dict
        Updated pyfhd configuration dictionary (possibly - check for any warnings in the log)
    """
    # Initialize cal dict
    cal = {}
    # Add any default values to calibration here
    cal["n_pol"] = min(obs["n_pol"], 2)
    cal["conv_thresh"] = pyfhd_config["cal_convergence_threshold"]
    cal["ref_antenna"] = 1
    cal["ref_antenna_name"] = obs["baseline_info"]["tile_names"][cal["ref_antenna"]]

    # Calculate auto-correlation visibilities, optionally use them for initial calibration estimates
    vis_auto, auto_tile_i = vis_extract_autocorr(obs, vis_arr, pyfhd_config)
    # Calculate auto-correlation visibilities
    vis_auto_model, auto_tile_i = vis_extract_autocorr(obs, vis_model_arr, pyfhd_config)
    # Initalize the gain
    if pyfhd_config["calibration_auto_initialize"]:
        cal["gain"] = vis_cal_auto_init(
            obs, cal, vis_arr, vis_model_arr, vis_auto, vis_auto_model, auto_tile_i
        )
    else:
        cal["gain"] = np.full(
            (cal["n_pol"], obs["n_freq"], obs["n_tile"]),
            pyfhd_config["cal_gain_init"],
            dtype=np.complex128,
        )

    # Do the calibration with vis_calibrate_subroutine
    logger.info("Gain initialized beginning vis_calibrate subroutine")
    cal = vis_calibrate_subroutine(
        vis_arr, vis_model_arr, vis_weights, obs, cal, params, pyfhd_config, logger
    )
    logger.info("Function vis_calibrate_subroutine has completed.")
    if pyfhd_config["flag_calibration"]:
        logger.info(
            "Flagging Calibration has been activated and calibration will now be flagged"
        )
        obs = vis_calibration_flag(obs, cal, pyfhd_config, logger)

    # Copy the cal structure with per-frequency gain solutions for future comparisons
    cal_base = deepcopy(cal)

    # Perform bandpass (amp + phase per fine freq) and polynomial fitting (low order amp + phase fit plus cable reflection fit)
    if pyfhd_config["bandpass_calibrate"]:
        logger.info("You have chosen to perform a bandpass calculation and calibration")
        if pyfhd_config["auto_ratio_calibration"]:
            cal, auto_ratio = cal_auto_ratio_divide(obs, cal, vis_auto, auto_tile_i)
        else:
            auto_ratio = None
        cal_bandpass, cal_remainder = vis_cal_bandpass(obs, cal, pyfhd_config, logger)

        if pyfhd_config["calibration_polyfit"]:
            logger.info(
                "You have selected to perform polynomial fits over the frequency band"
            )
            cal_polyfit, pyfhd_config = vis_cal_polyfit(
                obs, cal_remainder, auto_ratio, pyfhd_config, logger
            )
            # Replace vis_cal_combine with this line as the gain is the same size for polyfit and bandpass
            cal["gain"] = cal_polyfit["gain"] * cal_bandpass["gain"]
            for key in cal_polyfit:
                if key not in cal:
                    cal[key] = cal_polyfit[key]
        else:
            cal = cal_bandpass

        if pyfhd_config["auto_ratio_calibration"]:
            cal = cal_auto_ratio_remultiply(cal, auto_ratio, auto_tile_i)
    elif pyfhd_config["calibration_polyfit"]:
        cal, pyfhd_config = vis_cal_polyfit(cal, obs, None, pyfhd_config, logger)

    # Get the gain residuals
    if pyfhd_config["calibration_auto_fit"]:
        # Get amp from auto-correlation visibilities for plotting (or optionally for the calibration solution itself)
        cal_auto = vis_cal_auto_fit(obs, cal, vis_auto, vis_auto_model, auto_tile_i)
        # These subtractions replace vis_cal_subtract
        cal_res_gain = cal_base["gain"] - cal_auto["gain"]
    else:
        # These subtractions replace vis_cal_subtract
        cal_res_gain = cal_base["gain"] - cal["gain"]

    # If calibration_auto_fit was set then replace cal with cal_auto, usually for diagnostic purposes
    if pyfhd_config["calibration_auto_fit"]:
        cal = cal_auto
    # Apply Calibration
    logger.info("Applying the calibration")
    vis_cal, cal = vis_calibration_apply(
        vis_arr, obs, cal, vis_model_arr, vis_weights, logger
    )
    cal["gain_residual"] = cal_res_gain

    # Save the ratio and sigma average variance related to vis_cal
    logger.info("Saving the ratio and sigma average variance")
    if pyfhd_config["vis_baseline_hist"]:
        cal["vis_baseline_hist"] = vis_baseline_hist(
            obs, params, vis_cal, vis_model_arr
        )

    # Calculate statistics to put into the calibration dictionary for output purposes
    logger.info("Calculating statistics from calibration")
    nc_pol = min(obs["n_pol"], 2)
    cal_gain_avg = np.zeros(nc_pol)
    cal_res_avg = np.zeros(nc_pol)
    cal_res_restrict = np.zeros(nc_pol)
    cal_res_stddev = np.zeros(nc_pol)
    for pol_i in range(nc_pol):
        tile_use_i = np.where(obs["baseline_info"]["tile_use"])[0]
        freq_use_i = np.where(obs["baseline_info"]["freq_use"])[0]
        if tile_use_i.size == 0 or freq_use_i.size == 0:
            continue
        # Replaced extract_subarray with just the proper indexing
        gain_ref = cal["gain"][pol_i, freq_use_i, :][:, tile_use_i]
        gain_res = cal_res_gain[pol_i, freq_use_i][:, tile_use_i]
        cal_gain_avg[pol_i] = np.mean(np.abs(gain_ref))
        cal_res_avg[pol_i] = np.mean(np.abs(gain_res))
        res_mean = resistant_mean(np.abs(gain_res), 2)
        cal_res_restrict[pol_i] = res_mean
        cal_res_stddev[pol_i] = np.std(np.abs(gain_res))

    cal["mean_gain"] = cal_gain_avg
    cal["mean_gain_residual"] = cal_res_avg
    cal["mean_gain_restrict"] = cal_res_restrict
    cal["stddev_gain_residual"] = cal_res_stddev

    if pyfhd_config["calibration_plots"]:
        logger.info(
            f"Plotting the calibration solutions into {pyfhd_config['output_dir'] / 'plots' / 'calibration'}"
        )
        plot_cals(obs, cal, pyfhd_config)

    # Return the calibrated visibility array
    return vis_cal, cal, obs, pyfhd_config


def calibrate_qu_mixing(
    vis_arr: NDArray[np.complex128],
    vis_model_arr: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    obs: dict,
) -> float:
    """
    Solve for the degenerate phase between pseudo Q (YY - XX) and pseudo U (YX + XY) for the calibrated data and
    the simulated model separately, and return their difference. This difference represents the excess mixing
    angle between Q and U due to the instrumental beam not captured in a typical polarization-independent
    calibration.

    Parameters
    ----------
    vis_arr : np.ndarray
        Uncalibrated data visiblities
    vis_model_arr : np.ndarray
        Simulated model visibilites
    vis_weights : np.ndarray
        Weights (flags) of the visibilities
    obs : dict
        Observation metadata dictionary

    Returns
    -------
    stokes_mix_phase : float
        The excess mixing angle between Q and U from instrumental effects
    """

    n_freq = obs["n_freq"]
    # n_tile = obs['n_tile']
    n_time = obs["n_time"]
    # This should be number of baselines for one time step
    n_baselines = obs["n_baselines"]

    # reshape from (n_freq, n_baselines*n_times) to (n_freq, n_times, n_baselines). Turns out due to the row major vs col major difference
    # between IDL and python, this shape also changes
    new_shape = (n_freq, n_time, n_baselines)

    # Use the xx weightss (yy should be identical at this point)
    weights_use = np.reshape(vis_weights[0, :, :], new_shape)
    # carried over from FHD code - not sure this is necessary (maybe avoids NaNs?)
    weights_use = np.maximum(weights_use, np.zeros_like(weights_use))
    weights_use = np.minimum(weights_use, np.ones_like(weights_use))

    # Q = YY - XX for data
    pseudo_q = reshape_and_average_in_time(
        vis_arr[1, :, :] - vis_arr[0, :, :], n_freq, n_time, n_baselines, weights_use
    )

    # U = YX + XY for data
    pseudo_u = reshape_and_average_in_time(
        vis_arr[3, :, :] + vis_arr[2, :, :], n_freq, n_time, n_baselines, weights_use
    )

    # Q = YY - XX for model
    pseudo_q_model = reshape_and_average_in_time(
        vis_model_arr[1, :, :] - vis_model_arr[0, :, :],
        n_freq,
        n_time,
        n_baselines,
        weights_use,
    )

    # U = YX + XY for model
    pseudo_u_model = reshape_and_average_in_time(
        vis_model_arr[3, :, :] + vis_model_arr[2, :, :],
        n_freq,
        n_time,
        n_baselines,
        weights_use,
    )

    weights_t_avg = np.sum(weights_use, axis=1)

    i_use = np.nonzero(weights_t_avg)
    pseudo_u = np.squeeze(pseudo_u[i_use]).flatten()
    pseudo_q = np.squeeze(pseudo_q[i_use]).flatten()
    pseudo_q_model = np.squeeze(pseudo_q_model[i_use]).flatten()
    pseudo_u_model = np.squeeze(pseudo_u_model[i_use]).flatten()

    # LA_LEAST_SQUARES does not use double precision by default as such you will see differences.
    # Also LA_LEAST_SQUARES uses different method to numpy, generally LA_LEAST_SQUARES assumes the first matrix
    # has full rank, while numpy does not assume this.

    x = np.vstack([pseudo_u, np.ones(pseudo_u.size)]).T
    u_q_mix = np.linalg.lstsq(x, pseudo_q, rcond=None)[0]
    u_q_phase = np.arctan2(u_q_mix[0].imag, u_q_mix[0].real)

    x = np.vstack([pseudo_u_model, np.ones(pseudo_u_model.size)]).T
    u_q_mix_model = np.linalg.lstsq(x, pseudo_q_model, rcond=None)[0]
    u_q_phase_model = np.arctan2(u_q_mix_model[0].imag, u_q_mix_model[0].real)

    return u_q_phase_model - u_q_phase

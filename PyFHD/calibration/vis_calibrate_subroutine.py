import numpy as np
from numpy.typing import NDArray
import warnings
from logging import Logger
from PyFHD.calibration.calibration_utils import calculate_adaptive_gain
from PyFHD.pyfhd_tools.pyfhd_utils import weight_invert, histogram, idl_median


def vis_calibrate_subroutine(
    vis_arr: NDArray[np.complex128],
    vis_model_ptr: NDArray[np.complex128],
    vis_weight_ptr: NDArray[np.float64],
    obs: dict,
    cal: dict,
    params: dict,
    pyfhd_config: dict,
    logger: Logger,
    calibration_weights=False,
    no_ref_tile=False,
):
    """
    Perform a linear least-squares regression between the data visilbities and the simulated model
    visiblities to solve for an amplitude and phase for each tile for each frequency channel.

    Parameters
    ----------
    vis_arr : NDArray[np.complex128]
        Uncalibrated data visiblities
    vis_model_ptr : NDArray[np.complex128]
        Simulated model visibilites
    vis_weight_ptr : NDArray[np.float64]
        Weights (flags) of the visibilities
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    pyfhd_config : dict
        PyFHD's configuration dictionary containing all the options set for a PyFHD run
    calibration_weights : bool, optional
        Weight the visibilities at the minimum and maximum baseline by a soft taper for
        calibration purposes only, by default False
    no_ref_tile : bool, optional
        Do not reference calibration phases at this stage, by default False

    Returns
    -------
    cal : dict
        Calibration dictionary

    Raises
    ------
    ValueError
        Should almost never happen, only gets raised for max_cal_iter has been set to less than 5, however
        since it's currently hardcoded in here it's more of an exception for any developers of PyFHD
    """
    # Retrieve values from data structures
    # There is a few hardcoded values in here that were previously hardcoded in fhd_struct_init_cal
    # If you wish to change them, add them to the pyfhd_config through pyfhd_setup and the config in pyfhd.yaml
    reference_tile = 1
    min_baseline = obs["min_baseline"]
    max_baseline = obs["max_baseline"]
    dimension = obs["dimension"]
    elements = obs["elements"]
    min_cal_baseline = (
        max(pyfhd_config["min_cal_baseline"], obs["min_baseline"])
        if pyfhd_config["min_cal_baseline"]
        else obs["min_baseline"]
    )
    max_cal_baseline = (
        min(pyfhd_config["max_cal_baseline"], obs["max_baseline"])
        if pyfhd_config["max_cal_baseline"]
        else obs["max_baseline"]
    )
    # minimum number of calibration equations needed to solve for the gain of one baseline
    min_cal_solutions = 5
    # For record keeping
    cal["min_cal_solutions"] = min_cal_solutions
    # average the visibilities across time steps before solving for the gains
    time_average = pyfhd_config["cal_time_average"]
    # maximum iterations to perform for the linear least-squares solver
    max_cal_iter = pyfhd_config["max_cal_iter"]
    # Leave a warning if its less than 5 iterations, or an Error if its less than 1
    if max_cal_iter < 5:
        warnings.warn(
            "At Least 5 calibrations iterations is recommended.\nYou're currently using {} iterations".format(
                int(max_cal_iter)
            )
        )
    elif max_cal_iter < 1:
        raise ValueError(
            "max_cal_iter should be 1 or more. A max_cal_iter of 5 or more is recommended"
        )
    conv_thresh = pyfhd_config["cal_convergence_threshold"]
    use_adaptive_gain = pyfhd_config["cal_adaptive_calibration_gain"]
    base_gain = pyfhd_config["cal_base_gain"]
    # halt if the strict convergence is worse than most of the last x iterations
    divergence_history = 3
    # halt if the convergence gets significantly worse by a factor of x in one iteration
    divergence_factor = 1.5
    # For record keeping
    cal["divergence_history"] = 3
    cal["divergence_factor"] = 1.5
    n_pol = cal["n_pol"]
    n_freq = obs["n_freq"]
    n_tile = obs["n_tile"]
    n_time = obs["n_time"]
    # weights WILL be over-written! (Only for NAN gain solutions)
    vis_weight_ptr_use = vis_weight_ptr
    # tile_a & tile_b contribution indexed from 0
    tile_A_i = obs["baseline_info"]["tile_a"] - 1
    tile_B_i = obs["baseline_info"]["tile_b"] - 1
    freq_arr = obs["baseline_info"]["freq"]
    n_baselines = obs["n_baselines"]
    if pyfhd_config["cal_phase_fit_iter"]:
        phase_fit_iter = pyfhd_config["cal_phase_fit_iter"]
    else:
        # This gets set multiple times in FHD, by default it's 4, and is set in fhd_struct_init_cal
        phase_fit_iter = np.min([np.floor(max_cal_iter / 4), 4])
    kbinsize = obs["kpix"]

    cal["convergence"] = np.zeros([n_pol, n_freq, n_tile])
    cal["conv_iter"] = np.zeros([n_pol, n_freq, n_tile])
    cal["n_converged"] = np.zeros(n_pol)
    cal["tile_flag"] = np.zeros([n_tile], dtype=np.bool)
    for pol_i in range(n_pol):
        logger.info(
            f"Beginning Calibration for polarization {pol_i} ({obs['pol_names'][pol_i]})"
        )
        convergence = np.zeros((n_freq, n_tile))
        conv_iter_arr = np.zeros((n_freq, n_tile))
        # Want to ensure we're not affecting the current array till we overwrite it
        gain_arr = cal["gain"][pol_i].copy()

        # Average the visibilities over the time steps before solving for the gains solutions
        # This is not recommended, as longer baselines will be downweighted artifically.
        if time_average:
            # The visibilities have dimension nfreq x (n_baselines x n_time),
            # which can be reformed to nfreq x n_baselines x n_time
            tile_A_i = tile_A_i[0:n_baselines]
            tile_B_i = tile_B_i[0:n_baselines]
            # So IDL does reforms as REFORM(x, cols, rows, num_of_col_row_arrays)
            # Python is row-major, so we need to flip that shape that is used in REFORM
            # Although, to get the same results in the shape we want we need to do a few
            # transposes
            shape = np.array([n_time, n_baselines, n_freq])
            # Transpose here so it's in the same shape as IDL when reshaping, then transpose
            # back to get the shape we need to suit the shape of the rest of the arrays
            # I suspect this is because the reshape has no guarantee of the memory layout
            # at the end, which means we swap indexes of values that is not consistent with IDL
            vis_weight_use_reshaped = np.reshape(vis_weight_ptr_use[pol_i].T, shape).T
            vis_weight_use = np.maximum(vis_weight_use_reshaped, 0)
            vis_weight_use = np.minimum(vis_weight_use, 1)
            vis_model = np.reshape(vis_model_ptr[pol_i].T, shape).T
            vis_model = np.sum(vis_model * vis_weight_use, axis=2)
            vis_measured = np.reshape(vis_arr[pol_i].T, shape).T
            vis_avg = np.sum(vis_measured * vis_weight_use, axis=2)
            weight = np.sum(vis_weight_use, axis=2)

            kx_arr = params["uu"][0:n_baselines] / kbinsize
            ky_arr = params["vv"][0:n_baselines] / kbinsize
        else:
            # In the case of not using a time_average do the following setup instead for weight and vis_avg
            vis_weight_use = np.maximum(0, vis_weight_ptr_use[pol_i])
            vis_weight_use = np.minimum(vis_weight_use, 1)
            vis_model = vis_model_ptr[pol_i] * vis_weight_use
            vis_avg = vis_arr[pol_i] * vis_weight_use
            weight = vis_weight_use

            kx_arr = params["uu"] / kbinsize
            ky_arr = params["vv"] / kbinsize
        # Now use the common code from the two possibilities in vis_calibrate_subroutine.pro
        kr_arr = np.sqrt(kx_arr**2 + ky_arr**2)
        # When IDL does a matrix multiply on two 1D vectors it does the outer product.
        dist_arr = np.outer(kr_arr, freq_arr).T * kbinsize
        xcen = np.outer(abs(kx_arr), freq_arr).T
        ycen = np.outer(abs(ky_arr), freq_arr).T
        if calibration_weights:
            flag_dist_cut = np.where(
                (dist_arr < min_baseline)
                | (xcen > (elements / 2))
                | (ycen > (dimension / 2))
            )
            if min_cal_baseline > min_baseline:
                taper_min = np.maximum(
                    (np.sqrt(2) * min_cal_baseline - dist_arr) / min_cal_baseline, 0
                )
            else:
                taper_min = 0
            if max_cal_baseline < max_baseline:
                taper_max = np.maximum(
                    (dist_arr - max_cal_baseline) / min_cal_baseline, 0
                )
            else:
                taper_max = 0
            baseline_weights = np.maximum(1 - (taper_min + taper_max) ** 2, 0)
        else:
            flag_dist_cut = np.where(
                (dist_arr < min_cal_baseline)
                | (dist_arr > max_cal_baseline)
                | (xcen > (elements / 2))
                | (ycen > (dimension / 2))
            )
        # Remove kx_arr, ky_arr and dist_arr from the namespace, allow garbage collector to do its work
        del (kx_arr, ky_arr, dist_arr)

        if np.size(flag_dist_cut) > 0:
            weight[flag_dist_cut] = 0
        vis_avg *= weight_invert(weight)
        vis_model *= weight_invert(weight)

        tile_use_flag = obs["baseline_info"]["tile_use"]
        freq_use_flag = obs["baseline_info"]["freq_use"]

        freq_weight = np.sum(weight, axis=1)
        baseline_weight = np.sum(weight, axis=0)
        freq_use = np.where((freq_weight > 0) & (freq_use_flag > 0))[0]
        baseline_use = np.nonzero(baseline_weight)
        hist_tile_A, _, riA = histogram(tile_A_i[baseline_use], min=0, max=n_tile - 1)
        hist_tile_B, _, riB = histogram(tile_B_i[baseline_use], min=0, max=n_tile - 1)
        tile_use = np.where(((hist_tile_A + hist_tile_B) > 0) & (tile_use_flag > 0))[0]
        tile_flag = np.where(((hist_tile_A + hist_tile_B) == 0) & (tile_use_flag == 0))[
            0
        ]

        # Should be able to reduce precision if memory is a concern
        tile_A_i_use = np.zeros(np.size(baseline_use), dtype=np.int64)
        tile_B_i_use = np.zeros(np.size(baseline_use), dtype=np.int64)
        for tile_i in range(np.size(tile_use)):
            if hist_tile_A[tile_use[tile_i]] > 0:
                # Calculate tile contributions for each non-flagged baseline
                tile_A_i_use[riA[riA[tile_use[tile_i]] : riA[tile_use[tile_i] + 1]]] = (
                    tile_i
                )
            if hist_tile_B[tile_use[tile_i]] > 0:
                # Calculate tile contributions for each non-flagged baseline
                tile_B_i_use[riB[riB[tile_use[tile_i]] : riB[tile_use[tile_i] + 1]]] = (
                    tile_i
                )

        ref_tile_use = np.where(reference_tile == tile_use)
        if ref_tile_use[0].size == 0:
            ref_tile_use = 0
            cal["ref_antenna"] = tile_use[ref_tile_use]
            cal["ref_antenna_name"] = obs["baseline_info"]["tile_names"][
                cal["ref_antenna"]
            ]
        else:
            # Extract out value to avoid any weird stuff happening
            ref_tile_use = ref_tile_use[0][0]
        # Replace all NaNs with 0's
        vis_model[np.isnan(vis_model)] = 0

        conv_test = np.zeros((freq_use.size, max_cal_iter))
        n_converged = 0
        for fii in range(freq_use.size):
            fi = freq_use[fii]
            gain_curr = gain_arr[fi, tile_use]
            # Set up data and model arrays of the original and conjugated versions. This
            # provides twice as many equations into the linear least-squares solver.
            vis_data2 = np.squeeze(vis_avg[fi, baseline_use])
            vis_data2 = np.hstack([vis_data2, np.conj(vis_data2)])
            vis_model2 = np.squeeze(vis_model[fi, baseline_use])
            vis_model2 = np.hstack([vis_model2, np.conj(vis_model2)])
            weight2 = np.squeeze(weight[fi, baseline_use])
            weight2 = np.hstack([weight2, weight2])
            if calibration_weights:
                baseline_wts2 = np.squeeze(baseline_weights[fi, baseline_use])
                # Concatenate two of the same array, why?
                baseline_wts2 = np.hstack([baseline_wts2, baseline_wts2])

            b_i_use = np.where(weight2 > 0)
            weight2 = weight2[b_i_use]
            vis_data2 = vis_data2[b_i_use]
            vis_model2 = vis_model2[b_i_use]

            A_ind = np.hstack([tile_A_i_use, tile_B_i_use])
            A_ind = A_ind[b_i_use]
            B_ind = np.hstack([tile_B_i_use, tile_A_i_use])
            B_ind = B_ind[b_i_use]

            A_ind_arr = []
            n_arr = np.zeros(tile_use.size)
            for tile_i in range(tile_use.size):
                inds = np.where(A_ind == tile_i)[0]
                if inds.size > 1:
                    A_ind_arr.append(np.reshape(inds, (inds.size, 1)))
                else:
                    A_ind_arr.append(-1)
                # NEED SOMETHING MORE IN CASE INDIVIDUAL TILES ARE FLAGGED FOR ONLY A FEW FREQUENCIES!!
                n_arr[tile_i] = inds.size
            # I suspect a list of lists maybe faster than the object array, check during optimization
            # Although I doubt it will make a huge difference.
            A_ind_arr = np.array(A_ind_arr, dtype=object)
            # For tiles which don't satisfy the minimum number of solutions, pre-emptively set them to 0
            # in order to prevent certain failure in meeting strict convergence threshold
            inds_min_cal = np.where(n_arr < min_cal_solutions)[0]
            if inds_min_cal.size > 0:
                gain_curr[inds_min_cal] = 0
            gain_new = np.zeros(tile_use.size, dtype=np.complex128)
            convergence_list = np.zeros(max_cal_iter)
            conv_gain_list = np.zeros(max_cal_iter)
            convergence_loose = 0
            for i in range(max_cal_iter):
                convergence_loose_prev = convergence_loose
                divergence_flag = 0
                vis_use = vis_data2

                vis_model_matrix = vis_model2 * np.conj(gain_curr[B_ind])
                for tile_i in range(tile_use.size):
                    if n_arr[tile_i] >= min_cal_solutions:
                        if calibration_weights:
                            xmat = vis_model_matrix[(A_ind_arr[tile_i]).astype(int)]
                            # For some reason IDL multiplcation just allows two arrays of
                            # very dissimilar sizes to be multiplied by just ignoring everything
                            # after the index of the smallest one!?! Could be worth checking this is
                            # what this line was meant to do.
                            xmat = xmat.flatten()
                            xmat_dag = np.conj(xmat) * baseline_wts2[0 : xmat.size]
                            gain_new[tile_i] = (
                                1
                                / np.dot(xmat, xmat_dag)
                                * np.dot(
                                    np.transpose(
                                        vis_use[(A_ind_arr[tile_i]).astype(int)]
                                    ),
                                    xmat_dag,
                                )
                            )[0]
                        else:
                            gain_new[tile_i] = np.linalg.lstsq(
                                vis_model_matrix[(A_ind_arr[tile_i]).astype(int)],
                                vis_use[(A_ind_arr[tile_i]).astype(int)],
                                rcond=None,
                            )[0][0][0]

                gain_old = gain_curr.copy()
                if np.sum(np.abs(gain_new)) == 0:
                    gain_curr = gain_new.copy()
                    # Break the loop
                    break
                if phase_fit_iter - i > 0:
                    # fit only phase at first
                    gain_new *= np.abs(gain_old) * weight_invert(np.abs(gain_new))
                if use_adaptive_gain:
                    conv_gain = calculate_adaptive_gain(
                        conv_gain_list,
                        convergence_list,
                        i,
                        base_gain,
                        final_convergence_estimate=0,
                    )
                else:
                    conv_gain = base_gain
                gain_curr = (gain_new * conv_gain + gain_old * base_gain) / (
                    base_gain + conv_gain
                )
                dgain = np.abs(gain_curr) * weight_invert(np.abs(gain_old))
                diverge_i = np.where(dgain < np.abs(gain_old) / 2)[0]
                if diverge_i.size > 0:
                    gain_curr[diverge_i] = (
                        gain_new[diverge_i] + gain_old[diverge_i] * 2
                    ) / 3
                if np.size(np.where(np.isnan(gain_curr))) > 0:
                    gain_curr[np.where(np.isnan(gain_curr))] = gain_old[
                        np.where(np.isnan(gain_curr))
                    ]
                if not no_ref_tile:
                    gain_curr *= np.conj(gain_curr[ref_tile_use]) / np.abs(
                        gain_curr[ref_tile_use]
                    )
                convergence_strict = np.max(
                    np.abs(gain_curr - gain_old) * weight_invert(np.abs(gain_old))
                )
                convergence_loose = np.mean(
                    np.abs(gain_curr - gain_old) * weight_invert(np.abs(gain_old))
                )
                convergence_list[i] = convergence_strict
                conv_test[fii, i] = convergence_strict
                if i > phase_fit_iter + divergence_history:
                    if convergence_strict <= conv_thresh:
                        # Stop if the solution has converged to the specified threshold
                        n_converged += 1
                        break
                    if convergence_loose >= convergence_loose_prev:
                        # Stop if the solutions are no longer converging
                        if (convergence_loose <= conv_thresh) or (
                            convergence_loose_prev <= conv_thresh
                        ):
                            n_converged += 1
                            if convergence_loose <= conv_thresh:
                                # If the previous solution met the threshold, but the current one did not, then
                                # back up one iteration and use the previous solution
                                gain_curr = gain_old.copy()
                                convergence[fi, tile_use] = conv_test[fii, i - 1]
                                conv_iter_arr[fi, tile_use] = i - 1
                            break
                        else:
                            # Halt if the strict convergence is worse than most of the recent iterations
                            # Needed to use IDL_MEDIAN here as the numpy median was producing values different
                            # from IDL as the even keyword was not used. This need to may change over time as
                            # the calibration gets adjusted for double precision
                            divergence_test_1 = convergence_strict >= idl_median(
                                conv_test[fii, i - divergence_history - 1 : i]
                            )
                            # Also halt if the convergence gets significantly worse in one iteration
                            divergence_test_2 = (
                                convergence_strict
                                >= np.min(conv_test[fii, 0:i]) * divergence_factor
                            )
                            # possible bug fix; should we really test for divergence when only
                            # fitting the phase? Fix below doesn't use the phase-only portion
                            # of fitting when checking for divergence
                            # divergence_test_2 = convergence_strict >= np.min(conv_test[int(phase_fit_iter): i, fii]) * divergence_factor
                            if divergence_test_1 or divergence_test_2:
                                # If both measures of convergence are getting worse, we need to stop.
                                logger.info(
                                    f"Calibration diverged at iteration: {i}, for pol_i: {pol_i}, freq_i: {fi}. Convergence was: {conv_test[fii, i - 1]} and the threshold was: {conv_thresh}"
                                )
                                divergence_flag = True
                                break
            if divergence_flag:
                # If the solution diverged, back up one iteration and use the previous solution
                gain_curr = gain_old.copy()
                convergence[fi, tile_use] = conv_test[fii, i - 1]
                conv_iter_arr[fi, tile_use] = i - 1
            else:
                convergence[fi, tile_use] = np.abs(
                    gain_curr - gain_old
                ) * weight_invert(np.abs(gain_old))
                conv_iter_arr[fi, tile_use] = i
            if i == max_cal_iter:
                logger.info(
                    f"Calibration reach max iterations before converging for pol_i: {pol_i} and freq_i: {fi}. Convergence was: {conv_test[fii, i - 1]} and the threshold was: {conv_thresh}"
                )
            del A_ind_arr
            logger.info(
                f"Convergence was reached for polarization: {obs['pol_names'][pol_i]} ({pol_i}) and frequency: {fi}, with a convergence of: {conv_test[fii, i]} and the threshold was: {conv_thresh}"
            )
            gain_arr[fi, tile_use] = gain_curr
        nan_i = np.where(np.isnan(gain_curr))[0]
        if nan_i.size > 0:
            # any gains with NANs -> all tiles for that freq will have NANs
            freq_nan_i = nan_i % n_freq
            freq_nan_i = freq_nan_i[np.unique(freq_nan_i)]
            vis_weight_ptr_use[pol_i][:, freq_nan_i] = 0
            weight[:, freq_nan_i] = 0
            gain_arr[nan_i] = 0
        if tile_flag.size > 0:
            # set flagged tiles to NAN to remove from calculations
            gain_arr[:, tile_flag] = np.nan

        cal["tile_flag"][tile_flag] = True
        cal["gain"][pol_i] = gain_arr
        cal["convergence"][pol_i] = convergence
        cal["n_converged"][pol_i] = n_converged
        cal["conv_iter"][pol_i] = conv_iter_arr

    n_vis_cal = np.size(np.nonzero(weight)[0])
    cal["n_vis_cal"] = n_vis_cal
    return cal

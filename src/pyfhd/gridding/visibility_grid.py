import numpy as np
from numpy.typing import NDArray
from pyfhd.gridding.gridding_utils import (
    interpolate_kernel,
    baseline_grid_locations,
    grid_beam_per_baseline,
    conjugate_mirror,
)
from pyfhd.pyfhd_tools.pyfhd_utils import weight_invert, rebin, l_m_n, idl_argunique
from logging import Logger
import h5py


def visibility_grid(
    visibility: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    obs: dict,
    psf: dict | h5py.File,
    params: dict,
    polarization: int,
    pyfhd_config: dict,
    logger: Logger,
    uniform_flag: bool = False,
    no_conjugate: bool = False,
    model: NDArray[np.complex128] | None = None,
    fi_use: NDArray[np.integer] | None = None,
    bi_use: NDArray[np.integer] | None = None,
    verbose_logging=True,
) -> dict:
    """
    Put visibilities on a discrete, hyperresolved 2D plane in {u,v}-space with the Fourier-transform of the
    beam sensitivity as the kernel (or spreading function). This can done per frequency to create 3D {u,v,f}
    cubes that can generate power spectrum statistics, or this can be done once for all frequencies to create
    a single 2D {u,v} plane for continuum images. These {u,v} planes are the slant-orthographic projection of
    the sky when Fourier transformed.

    Gridding is done for calibrated data visibilities, (optionally) simulated model visibilities, weights, and
    variances. Weights and variances are crucial for propogating uncertainty estimates in the power spectrum
    space and for properly weighted images. The model is crucial for subtraction.

    The kernel is a extremely hyperresolved look-up table, which is (optionally) interpolated even further.
    Since the {u,v} pixels are discrete and the baseline locations are not, the kernel will populate the pixels
    in a unique way for each individual baseline. This code is optimized to provide the best estimate for each
    baseline whilst maintaining speed.

    Parameters
    ----------
    visibility : NDArray[np.complex128]
        Calibrated and flagged data visibilities
    vis_weights : NDArray[np.float64]
        Weights (flags) of the visibilities
    obs : dict
        Observation metadata dictionary
    psf : dict | h5py.File
        Beam metadata dictionary
    params : dict
        Visibility metadata dictionary
    polarization : int
        Index of the current polarization
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options for a run
    logger : Logger
        pyfhd's logger
    uniform_flag : bool, optional
        Grid a number count for contributing baselines per pixel, by default False
    no_conjugate : bool, optional
        Do not perform the conjugate mirror to fill half of the {u,v} plane, by default False
    model : NDArray[np.complex128] | None, optional
        Simulated model visibilites, by default None
    fi_use : NDArray[np.integer] | None, optional
        Frequency index array for gridding, i.e. gridding all frequencies for continuum images, by default None
    bi_use : NDArray[np.integer] | None, optional
        Baseline index array for gridding, i.e even vs odd time stamps, by default None
    verbose_logging : bool, optional
        If True, will log the gridding process, by default True
        Set to False if you're doing gridding per frequency (such as when creating HEALPIX files).

    Returns
    -------
    gridding_dict : dict
        A dictionary with all the gridded {u,v} planes, updated observation metadata dic, and the number of
        visibilties that where gridded.

    Raises
    ------
    ValueError
        Raised in the case the model provided was not a NumPy Array when a model is not None
    """

    # Get information from the data structures
    dimension = int(obs["dimension"])
    elements = int(obs["elements"])
    interp_flag = pyfhd_config["interpolate_kernel"]
    n_vis_arr = obs["nf_vis"]
    if fi_use is None:
        fi_use = np.nonzero(obs["baseline_info"]["freq_use"])[0]
    n_f_use = fi_use.size
    freq_bin_i = obs["baseline_info"]["fbin_i"]
    freq_bin_i = freq_bin_i[fi_use]

    # For each unflagged baseline, get the minimum contributing pixel number for gridding
    # and the 2D derivatives for bilinear interpolation
    baselines_dict = baseline_grid_locations(
        obs,
        psf,
        params,
        vis_weights,
        logger,
        bi_use=bi_use,
        fi_use=fi_use,
        mask_mirror_indices=pyfhd_config["mask_mirror_indices"],
        interp_flag=interp_flag,
    )
    # Extract data from the returned dictionary
    bin_n = baselines_dict["bin_n"]
    bin_i = baselines_dict["bin_i"]
    n_bin_use = baselines_dict["n_bin_use"]
    ri = baselines_dict["ri"]
    xmin = baselines_dict["xmin"]
    ymin = baselines_dict["ymin"]
    x_offset = baselines_dict["x_offset"]
    y_offset = baselines_dict["y_offset"]
    if bi_use is None:
        bi_use = baselines_dict["bi_use"]
    if interp_flag:
        dx0dy0_arr = baselines_dict["dx0dy0_arr"]
        dx0dy1_arr = baselines_dict["dx0dy1_arr"]
        dx1dy0_arr = baselines_dict["dx1dy0_arr"]
        dx1dy1_arr = baselines_dict["dx1dy1_arr"]

    # Instead of checking the visibilitity pointer we just take the vis_inds_use from visibility
    rows, cols = np.meshgrid(fi_use, bi_use)
    vis_arr_use = visibility[rows, cols].T
    # Model_flag has been removed in favor of just the model taking advantage that the model default is None
    # If it has been specified at all with anything other than None or False, then it should be a numpy array
    # if it isn't exit
    if model is not None:
        if isinstance(model, np.ndarray):
            model_use = model[rows, cols].T
            model_return = np.zeros((elements, dimension), dtype=np.complex128)
        else:
            raise ValueError(
                "Your model must be a numpy array when used as an argument"
            )

    # Now with the information we need, retrieve more data from the structures
    frequency_array = obs["baseline_info"]["freq"]
    frequency_array = frequency_array[fi_use]
    psf_dim = psf["dim"]
    psf_resolution = psf["resolution"]
    group_arr = psf["id"]
    if isinstance(psf, h5py.File):
        psf_dim = psf_dim[0]
        psf_resolution = psf_resolution[0]
        group_arr = group_arr[:]
    n_baselines = obs["n_baselines"]
    n_samples = obs["n_time"]
    # New group_arr code that is consistent with the FHD version
    # We go upto n_baselines in the case we have less baselines in the
    # observation than the transferred in beam. In most cases we're only
    # using one beam so this is fine, you might have issues if you're using
    # many beams across the baselines with this code. It passes the tests where
    # beams have been done per baseline so if you get an error, sorry!
    group_arr_baselines = np.min([n_baselines, group_arr.shape[-1]])
    group_arr = group_arr[polarization, freq_bin_i, :group_arr_baselines]
    # REBIN in IDL when expanding dimensions repeats 2D arrays after expanding
    group_arr = np.expand_dims(rebin(group_arr, (n_f_use, group_arr_baselines)), axis=0)
    group_arr = np.repeat(group_arr, n_samples, axis=0)
    group_arr = np.reshape(group_arr, (n_f_use, n_samples * group_arr_baselines))
    n_freq_use = frequency_array.size
    psf_dim2 = 2 * psf_dim
    psf_dim3 = psf_dim**2
    bi_use_reduced = bi_use % n_baselines

    # Flags have been defined in the function definition
    # Instead of reading the flags and then setting them.

    if pyfhd_config["beam_per_baseline"]:
        # Initialization for gridding operation via a low-res beam kernel, calculated per
        # baseline using offsets from image-space delays
        uu = params["uu"][bi_use]
        vv = params["vv"][bi_use]
        ww = params["ww"][bi_use]
        x = (np.arange(dimension) - dimension / 2) * obs["kpix"]
        y = x.copy()
        psf_intermediate_res = np.min(
            [np.ceil(np.sqrt(psf_resolution) / 2) * 2, psf_resolution]
        )
        psf_image_dim = psf["image_info"]["psf_image_dim"]
        if isinstance(psf_image_dim, h5py.Dataset):
            psf_image_dim = psf_image_dim[0]
        # psf_dim is guaranteed to be even
        image_bot = int(-(psf_dim / 2) * psf_intermediate_res + psf_image_dim / 2)
        image_top = int(
            (psf_dim * psf_resolution - 1)
            - (psf_dim / 2) * psf_intermediate_res
            + psf_image_dim / 2
        )
        l_mode, m_mode, n_tracked = l_m_n(obs, psf)
        n_tracked = np.zeros_like(n_tracked)

    # Initialize uv-arrays
    image_uv = np.zeros((elements, dimension), dtype=np.complex128)
    weights = np.zeros((elements, dimension), dtype=np.complex128)
    variance = np.zeros((elements, dimension))
    uniform_filter = np.zeros((elements, dimension))

    # If the uniform gridding has been activated we need to activate the uniform filter and switch off mapping if it has been activated
    if pyfhd_config["grid_uniform"]:
        uniform_flag = True

    conj_i = np.where(params["vv"][bi_use] > 0)[0]
    if conj_i.size > 0:
        if pyfhd_config["beam_per_baseline"]:
            uu[conj_i] = -uu[conj_i]
            vv[conj_i] = -vv[conj_i]
            ww[conj_i] = -ww[conj_i]
        vis_arr_use[:, conj_i] = np.conj(vis_arr_use[:, conj_i])
        if model is not None:
            model_use[:, conj_i] = np.conj(model_use[:, conj_i])

    # Return if all baselines have been flagged
    if n_bin_use == 0:
        logger.error("All data has been flagged")
        return {}

    n_vis = np.sum(bin_n)
    for fi in range(n_f_use):
        n_vis_arr[fi_use[fi]] = np.sum(xmin[fi, :] > 0)
    obs["nf_vis"] = n_vis_arr

    init_arr = np.zeros([psf_dim2, psf_dim2], dtype=np.complex128)

    arr_type = init_arr.dtype
    if pyfhd_config["grid_spectral"]:
        # Spectral B and Spectral D shouldn't reference each other just in case
        spectral_A = np.zeros([elements, dimension], dtype=np.complex128)
        spectral_B = np.zeros([elements, dimension])
        spectral_D = np.zeros([elements, dimension])
        if model is not None:
            spectral_model_A = np.zeros([elements, dimension], dtype=np.complex128)

    frequency_cache: dict[int, np.ndarray] = {}

    for bi in range(n_bin_use):
        # Cycle through sets of visibilities which contribute to the same data/model uv-plane pixels, and perform
        # the gridding operation per set using each visibilities' hyperresolved kernel

        # Select the indices of the visibilities which contribute to the same data/model uv-plane pixels
        inds = ri[ri[bin_i[bi]] : ri[bin_i[bi] + 1]]
        ind0 = inds[0]

        # Select the pixel offsets of the hyperresolution uv-kernel of the selected visibilities
        x_off = x_offset.flat[inds]
        y_off = y_offset.flat[inds]

        # Since all selected visibilities have the same minimum x,y pixel they contribute to,
        # reduce the array
        xmin_use = xmin.flat[ind0]
        ymin_use = ymin.flat[ind0]

        # Find the frequency group per index
        freq_i = inds % n_freq_use
        fbin = freq_bin_i[freq_i]

        # Calculate the number of selected visibilities and their baseline index
        vis_n = bin_n[bin_i[bi]]
        baseline_inds = bi_use_reduced[((inds / n_f_use) % n_baselines).astype(int)]

        if interp_flag:
            # Calculate the interpolated kernel on the uv-grid given the derivatives to baseline locations
            # and the hyperresolved pre-calculated beam kernel

            # Select the 2D derivatives to baseline locations
            dx1dy1 = dx1dy1_arr.flat[inds]
            dx1dy0 = dx1dy0_arr.flat[inds]
            dx0dy1 = dx0dy1_arr.flat[inds]
            dx0dy0 = dx0dy0_arr.flat[inds]

            # Select the model/data visibility values of the set, each with a weight of 1
            if model is not None:
                model_box = model_use.flat[inds]
            vis_box = vis_arr_use.flat[inds]
            psf_weight = np.ones(vis_n)

            box_matrix = np.zeros((vis_n, psf_dim3), dtype=arr_type)
            for ii in range(vis_n):
                if fbin[ii] not in frequency_cache:
                    to_interp = psf["beam_ptr"][polarization, fbin[ii]]
                    frequency_cache[fbin[ii]] = to_interp
                else:
                    to_interp = frequency_cache[fbin[ii]]
                # For each visibility, calculate the kernel values on the static uv-grid given the
                # hyperresolved kernel and an interpolation involving the derivatives
                box_matrix[ii] = interpolate_kernel(
                    to_interp,
                    x_off[ii],
                    y_off[ii],
                    dx0dy0[ii],
                    dx1dy0[ii],
                    dx0dy1[ii],
                    dx1dy1[ii],
                )
        else:
            # Calculate the beam kernel at each baseline location given the hyperresolved pre-calculated
            # beam kernel

            # Calculate a unique index for each kernel location and kernel type in order to reduce
            # operations if there are repeats
            group_id = group_arr.flat[inds]
            group_max = np.max(group_id) + 1
            xyf_i = (
                x_off + y_off * psf_resolution + fbin * psf_resolution**2
            ) * group_max + group_id

            # Calculate the unique number of kernel locations/types
            xyf_si = xyf_i.argsort(kind="stable")
            xyf_i = np.sort(xyf_i)
            xyf_ui = idl_argunique(xyf_i)
            n_xyf_bin = xyf_ui.size

            # There might be a better selection criteria to determine which is most efficient
            if vis_n > 1.1 * n_xyf_bin and not pyfhd_config["beam_per_baseline"]:
                # If there are any baselines which use the same beam kernel and the same discretized location
                # given the hyperresolution, then reduce the number of gridding operations to only
                # non-repeated baselines
                inds = inds[xyf_si]
                inds_use = xyf_si[xyf_ui]
                freq_i = freq_i[inds_use]

                x_off = x_off[inds_use]
                y_off = y_off[inds_use]
                fbin = fbin[inds_use]
                baseline_inds = baseline_inds[inds_use]
                if n_xyf_bin > 1:
                    xyf_ui0 = np.insert(xyf_ui[0 : n_xyf_bin - 1] + 1, 0, 0)
                else:
                    xyf_ui0 = np.array([0])
                psf_weight = xyf_ui - xyf_ui0 + 1

                vis_box1 = vis_arr_use.flat[inds]
                vis_box = vis_box1.flat[xyf_ui]
                if model is not None:
                    model_box1 = model_use.flat[inds]
                    model_box = model_box1.flat[xyf_ui]

                # For the baselines which map to the same pixels and use the same beam,
                # add the underlying data/model pixels such that the gridding operation
                # only needs to be performed once for the set
                if xyf_ui0.size == 1:
                    repeat_i = np.array([0])
                else:
                    repeat_i = np.where(psf_weight > 1)[0]
                xyf_ui = xyf_ui[repeat_i]
                xyf_ui0 = xyf_ui0[repeat_i]
                for rep_ii in range(repeat_i.size):
                    vis_box[repeat_i[rep_ii]] = np.sum(
                        vis_box1[xyf_ui0[rep_ii] : xyf_ui[rep_ii] + 1]
                    )
                    if model is not None:
                        model_box[repeat_i[rep_ii]] = np.sum(
                            model_box1[xyf_ui0[rep_ii] : xyf_ui[rep_ii] + 1]
                        )
                vis_n = n_xyf_bin
            else:
                # If there are not enough baselines which use the same beam kernel and discretized
                # location to warrent reduction, then perform the gridding operation per baseline
                if model is not None:
                    model_box = model_use.flat[inds]
                vis_box = vis_arr_use.flat[inds]
                psf_weight = np.ones(vis_n)
                # IDL had integer / integer i.e. 2015 / 336 == 5, used flooring divider instead in python
                # ALso do take note that each were very close always within 0.01 of their next number.
                # e.g. 2015 / 336 = 5.99, should ceiling be be used instead?
                bt_index = inds // n_freq_use

            box_matrix = np.zeros((vis_n, psf_dim3), dtype=arr_type)
            if pyfhd_config["beam_per_baseline"]:
                # Make the beams on the fly with corrective phases given the baseline location for each visibility
                # to the static uv-grid
                box_matrix = grid_beam_per_baseline(
                    psf,
                    pyfhd_config,
                    logger,
                    uu,
                    vv,
                    ww,
                    l_mode,
                    m_mode,
                    n_tracked,
                    frequency_array,
                    x,
                    y,
                    xmin_use,
                    ymin_use,
                    freq_i,
                    bt_index,
                    polarization,
                    image_bot,
                    image_top,
                    psf_dim3,
                    box_matrix,
                    vis_n,
                )
            else:
                for ii in range(vis_n):
                    if fbin[ii] not in frequency_cache:
                        box_mat = psf["beam_ptr"][polarization, fbin[ii]]
                        frequency_cache[fbin[ii]] = box_mat
                    else:
                        box_mat = frequency_cache[fbin[ii]]
                    # For each visibility, calculate the kernel values on the static uv-grid given the
                    # hyperresolved kernel
                    box_matrix[ii, :] = box_mat[x_off[ii], y_off[ii]]

        #  Calculate the conjugate transpose (dagger) of the uv-pixels that the current beam kernel contributes to
        box_matrix_dag = np.conj(box_matrix)

        if pyfhd_config["grid_spectral"]:
            term_A_box = np.dot(
                np.transpose(box_matrix_dag), np.transpose((freq_i * vis_box) / n_vis)
            )
            term_B_box = np.dot(
                np.transpose(box_matrix_dag), np.transpose(freq_i / n_vis)
            )
            term_D_box = np.dot(
                np.transpose(box_matrix_dag), np.transpose(freq_i**2 / n_vis)
            )
            spectral_A[
                xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
            ].flat += term_A_box
            spectral_B[
                xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
            ].flat += term_B_box.real
            spectral_D[
                xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
            ].flat += term_D_box.real
            # del(term_A_box, term_B_box, term_D_box)
            if model is not None:
                term_Am_box = np.dot(
                    np.transpose(box_matrix_dag),
                    np.transpose((freq_i * model_box) / n_vis),
                )
                spectral_model_A[
                    xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
                ] += term_Am_box

        if model is not None:
            # If model visibilities are being gridded, calculate the product of the model vis and the beam kernel
            # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane

            # Ensure model_box is flat, sometimes odd shapes can come in from metadata
            model_box = model_box.flatten()
            box_arr = np.dot(
                np.transpose(box_matrix_dag), np.transpose(model_box / n_vis)
            )
            model_return[
                xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
            ].flat += box_arr

        # Calculate the product of the data vis and the beam kernel
        # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane

        # Ensure vis_box is flat, sometimes odd shapes can come in from metadata
        vis_box = vis_box.flatten()
        box_arr = np.dot(np.transpose(box_matrix_dag), vis_box / n_vis)
        image_uv[
            xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
        ].flat += box_arr
        del box_arr

        if pyfhd_config["grid_weights"]:
            # If weight visibilities are being gridded, calculate the product the weight (1 per vis) and the beam kernel
            # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane
            wts_box = np.dot(
                np.transpose(box_matrix_dag), np.transpose(psf_weight / n_vis)
            )
            weights[
                xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
            ].flat += wts_box

        if pyfhd_config["grid_variance"]:
            # If variance visibilities are being gridded, calculate the product the weight (1 per vis) and the square
            # of the beam kernel for all vis which contribute to the same static uv-pixels, and add to the static uv-plane
            var_box = np.dot(
                np.transpose(np.abs(box_matrix_dag) ** 2),
                np.transpose(psf_weight / n_vis),
            )
            variance[
                xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
            ].flat += var_box

        if uniform_flag:
            uniform_filter[
                xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
            ] += bin_n[bin_i[bi]]

        if verbose_logging and (
            (n_bin_use <= 10)
            or (bi in np.arange(n_bin_use // 10, n_bin_use, n_bin_use // 10))
        ):
            logger.info(
                f"Gridding visibilities for baseline {bi} of {n_bin_use} for polarization {obs['pol_names'][polarization]}"
            )

    # Free Up Memory
    del (
        vis_arr_use,
        xmin,
        ymin,
        ri,
        inds,
        x_offset,
        y_offset,
        bin_i,
        bin_n,
        frequency_cache,
    )

    if model is not None:
        del model_use

    # Option to use spectral index information to scale the uv-plane
    if pyfhd_config["grid_spectral"]:
        spectral_uv = (spectral_A - n_vis * spectral_B * image_uv) * weight_invert(
            spectral_D - spectral_B**2
        )
        if model is not None:
            spectral_model_uv = (
                spectral_model_A - n_vis * spectral_B * model_return
            ) * weight_invert(spectral_D - spectral_B**2)
        if not no_conjugate:
            spectral_uv = (spectral_uv + conjugate_mirror(spectral_uv)) / 2
            if model is not None:
                spectral_model_uv = (
                    spectral_model_uv + conjugate_mirror(spectral_model_uv)
                ) / 2

    # Option to apply a uniform weighted filter to all uv-planes
    if pyfhd_config["grid_uniform"]:
        filter_use = weight_invert(uniform_filter, threshold=1)
        wts_i = np.nonzero(filter_use)
        if wts_i[0].size > 0:
            filter_use /= np.mean(filter_use[wts_i])
        else:
            filter_use /= np.mean(filter_use)
        image_uv *= weight_invert(filter_use)
        if pyfhd_config["grid_weights"]:
            weights *= weight_invert(filter_use)
        if pyfhd_config["grid_variance"]:
            variance *= weight_invert(filter_use)
        if model is not None:
            model_return *= weight_invert(filter_use)

    if not no_conjugate:
        # The uv-plane is its own conjugate mirror about the x-axis, so fill in the rest of the uv-plane
        # using simple maths instead of extra gridding
        image_uv = (image_uv + conjugate_mirror(image_uv)) / 2
        if pyfhd_config["grid_weights"]:
            weights = (weights + conjugate_mirror(weights)) / 2
        if pyfhd_config["grid_variance"]:
            variance = (variance + conjugate_mirror(variance)) / 4
        if model is not None:
            model_return = (model_return + conjugate_mirror(model_return)) / 2
        if uniform_flag:
            uniform_filter = (uniform_filter + conjugate_mirror(uniform_filter)) / 2

    # Arrange returns into a dictionary
    gridding_dict = {
        "image_uv": image_uv,
        "weights": weights,
        "variance": variance,
        "uniform_filter": uniform_filter,
        "obs": obs,
        "n_vis": n_vis,
    }

    if pyfhd_config["grid_spectral"]:
        gridding_dict["spectral_uv"] = spectral_uv

    if model is not None:
        gridding_dict["model_return"] = model_return
        if pyfhd_config["grid_spectral"]:
            gridding_dict["spectral_model_uv"] = spectral_model_uv

    return gridding_dict

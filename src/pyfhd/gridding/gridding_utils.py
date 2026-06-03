import numpy as np
from numpy.typing import NDArray
import pyfhd.gridding.filters as filters
from pyfhd.pyfhd_tools.pyfhd_utils import rebin, histogram, array_match, meshgrid
from scipy.signal import convolve
from astropy.convolution import Box2DKernel
from math import pi
from logging import Logger
import h5py


def interpolate_kernel(
    kernel_arr: NDArray[np.complex128],
    x_offset: NDArray[np.integer],
    y_offset: NDArray[np.integer],
    dx0dy0: NDArray[np.float64],
    dx1dy0: NDArray[np.float64],
    dx0dy1: NDArray[np.float64],
    dx1dy1: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Perform a bilinear interpolation given the 2D derivatives of the baseline center to the nearest
    hyperresolved pixel edge in {u,v} space. This will provide a quadratic estimate at the sample
    location to smooth out the dependence on hyperresolved pixel size.

    Parameters
    ----------
    kernel_arr : NDArray[np.complex128]
        The 2D kernel for bilinear interpolation
    x_offset : NDArray[np.integer]
        The nearest pixel offset in the x (u) direction
    y_offset : NDArray[np.integer]
        The nearest pixel offset in the y (v) direction
    dx0dy0 : NDArray[np.float64]
       (1 - derivative to pixel edge in x) * (1 - derivative to pixel edge in y) for selected baselines
    dx1dy0 : NDArray[np.float64]
        (derivative to pixel edge in x) * (1 - derivative to pixel edge in y) for selected baselines
    dx0dy1 : NDArray[np.float64]
        (1 - derivative to pixel edge in x) * (derivative to pixel edge in y) for selected baselines
    dx1dy1 : NDArray[np.float64]
        (derivative to pixel edge in x) * (derivative to pixel edge in y) for selected baselines

    Returns
    -------
    kernel: NDArray[np.complex128]
        The interpolated 2D kernel
    """
    kernel = kernel_arr[x_offset, y_offset] * dx0dy0
    kernel += kernel_arr[x_offset + 1, y_offset] * dx1dy0
    kernel += kernel_arr[x_offset, y_offset + 1] * dx0dy1
    kernel += kernel_arr[x_offset + 1, y_offset + 1] * dx1dy1

    return kernel


def conjugate_mirror(
    image: NDArray[np.complex128 | np.float64],
) -> NDArray[np.complex128 | np.float64]:
    """
    Mirror an image about the origin and take the complex conjugate. The origin is considered to be the first
    row, and thus is not repeated in the conjugate mirror.

    Parameters
    ----------
    image: NDArray[np.complex128 | np.float64]
        A 2D array of real or complex numbers

    Returns
    -------
    conj_mirror_image: NDArray[np.complex128 | np.float64]
        The 2D conjugate mirror of the input without the origin
    """
    # Flip image left to right (i.e. flips columns) & Flip image up to down (i.e. flips rows)
    conj_mirror_image = np.flip(image)
    # Shifts columns then rows by 1
    conj_mirror_image = np.roll(conj_mirror_image, 1, axis=(1, 0))
    # If any of the array is complex, or its a complex array, get the conjugates
    if np.iscomplexobj(image):
        conj_mirror_image = np.conjugate(conj_mirror_image)
    return conj_mirror_image


def baseline_grid_locations(
    obs: dict,
    psf: dict,
    params: dict,
    vis_weights: NDArray[np.float64],
    logger: Logger,
    bi_use: NDArray[np.integer] | None = None,
    fi_use: NDArray[np.integer] | None = None,
    fill_model_visibilities: bool = False,
    interp_flag: bool = False,
    mask_mirror_indices: bool = False,
) -> dict:
    """
    Calculate the histogram of baseline grid locations in units of pixels whilst also
    returning the minimum pixel number that an unflagged baseline contributes to (depending on the
    size of the kernel). Optionally return the 2D derivatives for bilinear interpolation and the
    indices of the unflagged baselines/frequencies.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    psf : dict
        Beam metadata dictionary
    params : dict
        Visibility metadata dictionary
    vis_weights : NDArray[np.float64]
        Weights (flags) of the visibilities
    logger : Logger
        pyfhd's logger
    bi_use : NDArray[np.integer] | None, optional
        Baseline index array for gridding, i.e even vs odd time stamps, by default None
    fi_use : NDArray[np.integer] | None, optional
        Frequency index array for gridding, i.e. gridding all frequencies for continuum images, by default None
    fill_model_visibilities : bool, optional
        Calculate baseline grid locations for all baselines, regardless of flags, by default False
    interp_flag : bool, optional
        Calculate derivatives for bilinear interpolation of the kernel to pixel locations, by default False
    mask_mirror_indices : bool, optional
        Exclude baselines mirrored along the v-axis, by default False

    Returns
    -------
    baselines_dict : dict
        Histogram of baseline grid locations, associated derivatives, and minimum contributing pixel location,
        arranged in a dictionary
    """

    # Set up the return dictionary
    baselines_dict = {}

    # Retrieve information from the data structures
    n_tile = obs["n_tile"]
    n_freq = obs["n_freq"]
    dimension = obs["dimension"]
    elements = obs["elements"]
    kbinsize = obs["kpix"]
    min_baseline = obs["min_baseline"]
    max_baseline = obs["max_baseline"]
    b_info = obs["baseline_info"]
    psf_dim = psf["dim"]
    psf_resolution = psf["resolution"]
    if isinstance(psf, h5py.File):
        psf_dim = psf_dim[0]
        psf_resolution = psf_resolution[0]

    # Frequency information of the visibilities
    if fill_model_visibilities:
        fi_use = np.arange(n_freq)
    elif fi_use is None:
        fi_use = np.nonzero(b_info["freq_use"])[0]
    frequency_array = b_info["freq"]
    frequency_array = frequency_array[fi_use]

    # Set the bi_use_flag, if we set it here, we need to return it
    bi_use_flag = False
    # Baselines to use
    if bi_use is None:
        bi_use_flag = True
        # if the data is being gridded separately for the even/odd time samples
        # then force flagging to be consistent across even/odd sets
        if not fill_model_visibilities:
            flag_test = np.sum(np.maximum(vis_weights, 0), axis=0)
            bi_use = np.nonzero(flag_test)[0]
        else:
            tile_use = np.arange(n_tile) + 1
            bi_use = array_match(
                b_info["tile_a"].astype(int),
                tile_use,
                array_2=b_info["tile_b"].astype(int),
            )

    # Rather than calculating the flat indexes we want, lets just index the array
    # by the frequency use and baseline_use indexes
    rows, cols = np.meshgrid(fi_use, bi_use)
    vis_weights_use = vis_weights[rows, cols].T

    # Units in pixel/Hz
    kx_arr = params["uu"][bi_use] / kbinsize
    ky_arr = params["vv"][bi_use] / kbinsize

    if not fill_model_visibilities:
        # Flag baselines on their maximum and minimum extent in the full frequency range of the observation
        # This prevents the sudden dissapearance of baselines along frequency
        dist_test = np.sqrt(kx_arr**2 + ky_arr**2) * kbinsize
        dist_test_max = np.max(obs["baseline_info"]["freq"]) * dist_test
        dist_test_min = np.min(obs["baseline_info"]["freq"]) * dist_test
        flag_dist_baseline = np.where(
            (dist_test_min < min_baseline) | (dist_test_max > max_baseline)
        )
        del (dist_test, dist_test_max, dist_test_min)

    # Create the other half of the uv plane via negating the locations
    conj_i = np.where(ky_arr > 0)[0]
    if conj_i.size > 0:
        kx_arr[conj_i] = -kx_arr[conj_i]
        ky_arr[conj_i] = -ky_arr[conj_i]

    # Center of baselines for x and y in units of pixels
    xcen = np.outer(frequency_array, kx_arr)
    ycen = np.outer(frequency_array, ky_arr)

    # Pixel number offset per baseline for each uv-box subset
    x_offset = np.fix(
        np.floor((xcen - np.floor(xcen)) * psf_resolution) % psf_resolution
    ).astype(np.int64)
    y_offset = np.fix(
        np.floor((ycen - np.floor(ycen)) * psf_resolution) % psf_resolution
    ).astype(np.int64)

    if interp_flag:
        # Derivatives from pixel edge to baseline center for use in interpolation
        dx_arr = (xcen - np.floor(xcen)) * psf_resolution - np.floor(
            (xcen - np.floor(xcen)) * psf_resolution
        )
        dy_arr = (ycen - np.floor(ycen)) * psf_resolution - np.floor(
            (ycen - np.floor(ycen)) * psf_resolution
        )
        baselines_dict["dx0dy0_arr"] = (1 - dx_arr) * (1 - dy_arr)
        baselines_dict["dx0dy1_arr"] = (1 - dx_arr) * dy_arr
        baselines_dict["dx1dy0_arr"] = dx_arr * (1 - dy_arr)
        baselines_dict["dx1dy1_arr"] = dx_arr * dy_arr

    # The minimum pixel in the uv-grid (bottom left of the kernel) that each baseline contributes to
    xmin = (np.floor(xcen) + elements / 2 - (psf_dim / 2 - 1)).astype(np.int64)
    ymin = (np.floor(ycen) + dimension / 2 - (psf_dim / 2 - 1)).astype(np.int64)

    # Set the minimum pixel value of baselines which fall outside of the uv-grid tom -1 to exclude them
    range_test_x_i = np.where((xmin <= 0) | (xmin + psf_dim - 1 >= elements - 1))
    range_test_y_i = np.where((ymin <= 0) | (ymin + psf_dim - 1 >= dimension - 1))
    if range_test_x_i[0].size > 0:
        xmin[range_test_x_i] = -1
        ymin[range_test_x_i] = -1
    if range_test_y_i[0].size > 0:
        xmin[range_test_y_i] = -1
        ymin[range_test_y_i] = -1

    # Flag baselines which fall outside the uv plane
    if not fill_model_visibilities:
        if np.size(flag_dist_baseline) > 0:
            # If baselines fall outside the desired min/max baseline range at all during the frequency range
            # then set their maximum pixel value to -1 to exclude them
            xmin[:, flag_dist_baseline] = -1
            ymin[:, flag_dist_baseline] = -1
            del flag_dist_baseline

    # Normally we check vis_weight_switch, but its always true here so... do this
    flag_i = np.where(vis_weights_use <= 0)
    if fill_model_visibilities:
        n_flag = 0
    else:
        n_flag = np.size(flag_i)
    if n_flag > 0:
        xmin[flag_i] = -1
        ymin[flag_i] = -1

    if mask_mirror_indices:
        # Option to exclude v-axis mirrored baselines
        if conj_i.size > 0:
            xmin[:, conj_i] = -1
            ymin[:, conj_i] = -1

    # If xmin or ymin is invalid then adjust the baselines dict as necessary
    if xmin.size == 0 or ymin.size == 0 or np.max([np.max(xmin), np.max(ymin)]) < 0:
        logger.warning("All data flagged or cut!")
        baselines_dict["bin_n"] = 0
        baselines_dict["n_bin_use"] = 0
        baselines_dict["bin_i"] = -1
        baselines_dict["ri"] = 0
    else:
        # Match all visibilities that map from and to exactly the same pixels and store them as a histogram in bin_n
        # with their respective index ri. Setting min equal to 0, excludes flagged data (data set to -1).
        for_hist = xmin + ymin * dimension
        bin_n, _, ri = histogram(for_hist, min=0)
        bin_i = np.nonzero(bin_n)[0]

        # Update the baselines_dict which gets returned
        baselines_dict["bin_n"] = bin_n
        baselines_dict["bin_i"] = bin_i
        baselines_dict["n_bin_use"] = bin_i.size
        baselines_dict["ri"] = ri
    # Add values to baselines dict
    baselines_dict["xmin"] = xmin
    baselines_dict["ymin"] = ymin
    baselines_dict["x_offset"] = x_offset
    baselines_dict["y_offset"] = y_offset
    if bi_use_flag:
        baselines_dict["bi_use"] = bi_use

    # Return the dictionary
    return baselines_dict


def dirty_image_generate(
    dirty_image_uv: NDArray[np.complex128],
    pyfhd_config: dict,
    logger: Logger,
    uniform_filter_uv: NDArray[np.float64] | None = None,
    mask: NDArray[np.integer] | None = None,
    baseline_threshold: int | float | None = None,
    normalization: float | NDArray[np.float64] | None = None,
    resize: int | None = None,
    width_smooth: int | float | None = None,
    degpix: float | None = None,
    not_real: bool = False,
    pad_uv_image: int | float | None = None,
    weights: NDArray[np.float64] | None = None,
    filter: NDArray[np.float64] | None = None,
    beam_ptr: NDArray[np.complex128] | None = None,
) -> tuple[
    NDArray[np.complex128], NDArray[np.float64], float | NDArray[np.float64] | None
]:
    """
    Generate a projected image from an input {u,v} plane through a 2D FFT. Optionally apply padding, masking,
    filtering, and more.

    Parameters
    ----------
    dirty_image_uv : NDArray[np.complex128]
        A 2D {u,v} plane which generally includes the beam via a gridding kernel
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger
    uniform_filter_uv : NDArray[np.float64] | None, optional
        A 2D {u,v} gridded visibility number, by default None
    mask : NDArray[np.integer] | None, optional
        A 2D {u,v} mask to apply before image creation, by default None
    baseline_threshold : int | float, optional
        The maximum baseline length to include in units of pixels, default None
    normalization : float | NDArray[np.float64] | None, optional
        A value by which to normalize the image by, by default None
    resize : int | None, optional
        Increase the number of pixels by a factor and rebin the input {u,v} plane, by default None
    width_smooth : int | float | None, optional
        Smooth out the harsh baseline threshold by the given number of pixels, by default None
    degpix : float | None, optional
        Degrees per pixel, by default None
    not_real : bool, optional
        Flag to return a complex image, by default False
    pad_uv_image : int | float | None, optional
        Pad the {u,v} plane by this pixel amount to increase perceived resolution of the image, by default None
    weights : NDArray[np.float64] | None, optional
        Gridded {u,v} plane of visibility weights, necessary in some filtering schemes, by default None
    filter : NDArray[np.float64] | None, optional
        Image filter to apply, by default None
    beam_ptr : NDArray[np.complex128] | None, optional
        Weight by an additional factor of the beam for optimal weighting, by default None

    Returns
    -------
    dirty_image : NDArray[np.complex128]
        A 2D {l,m} directional-cosine image plane
    filter : NDArray[np.float64]
        The filter applied to the dirty image
    normalization : float | NDArray[np.float64] | None
        The normalization (if any) applied to the dirty image
    """

    # dimension is columns, elements is rows
    elements, dimension = dirty_image_uv.shape
    di_uv_use = dirty_image_uv
    # If the baseline threshold has been set
    if baseline_threshold is not None:
        if width_smooth is None:
            width_smooth = np.floor(np.sqrt(dimension * elements) / 100)
        rarray = np.sqrt(
            (meshgrid(dimension, 1) - dimension / 2) ** 2
            + (meshgrid(elements, 2) - elements / 2) ** 2
        )
        # Get all the values that meet the threshold
        if baseline_threshold >= 0:
            cut_i = np.where(rarray.flatten() < baseline_threshold)
        else:
            cut_i = np.where(rarray.flatten() > np.abs(baseline_threshold))
        # Create the mask array of ones
        mask_bt = np.ones((elements, dimension))
        # If there are values from cut, then use all those here and replace with 0
        if np.size(cut_i) > 0:
            mask_bt_flatiter = mask_bt.flat
            mask_bt_flatiter[cut_i] = 0
        # Get the kernel width
        kernel_width = np.max([width_smooth, 1])
        # In IDL if the kernel width is even one is added to make it odd
        if kernel_width % 2 == 0:
            kernel_width += 1
        # Use a box width averaging filter over the mask, use valid so we can insert it in later
        box_averages = convolve(mask_bt, Box2DKernel(kernel_width), mode="valid")
        # Since IDL SMOOTH edges by default are the edges of the array used, ignore edges (its the reason why we used a valid convolve)
        start = int(kernel_width // 2)
        end = int(mask_bt.shape[1] - (kernel_width // 2))
        mask_bt[start:end, start:end] = box_averages
        # Apply boxed mask to the dirty image
        di_uv_use *= mask_bt

    # If a mask was supplied use that too
    if mask is not None:
        di_uv_use *= mask

    # If a filter was supplied as a numpy array (we can adjust this to support different formats)
    if filter is not None:
        if isinstance(filter, np.ndarray):
            # If the filter is already the right size, use it
            if np.size(filter) == np.size(di_uv_use):
                di_uv_use *= filter
            # Otherwise use a filter function
            else:
                if pyfhd_config["image_filter"] == "filter_uv_uniform":
                    logger.info("Using filter_uv_uniform for dirty_image_generate")
                elif pyfhd_config["image_filter"] == "filter_uv_hannning":
                    logger.warning(
                        "filter_uv_hanning hasn't been translated yet using filter_uv_uniform for dirty_image_generate instead"
                    )

                elif pyfhd_config["image_filter"] == "filter_uv_natural":
                    logger.warning(
                        "filter_uv_natural hasn't been translated yet using filter_uv_uniform for dirty_image_generate instead"
                    )

                elif pyfhd_config["image_filter"] == "filter_uv_radial":
                    logger.warning(
                        "filter_uv_radial hasn't been translated yet using filter_uv_uniform for dirty_image_generate instead"
                    )

                elif pyfhd_config["image_filter"] == "filter_uv_tapered_uniform":
                    logger.warning(
                        "filter_uv_tapered_uniform hasn't been translated yet using filter_uv_uniform for dirty_image_generate instead"
                    )

                elif pyfhd_config["image_filter"] == "filter_uv_optimal":
                    logger.warning(
                        "filter_uv_optimal hasn't been translated yet using filter_uv_uniform for dirty_image_generate instead"
                    )
                # Since we only use filter_uniform at the moment, put the call to it here.
                di_uv_use, filter = filters.filter_uv_uniform(
                    di_uv_use, vis_count=uniform_filter_uv, weights=weights
                )

    # Resize the dirty image by the factor resize
    if resize is not None:
        dimension *= resize
        elements *= resize
        # Ensure elements and dimension are integers as resize should be an int
        # but if it's not this makes sure dimension and elements are ints afterwards
        dimension = int(dimension)
        elements = int(elements)
        di_uv_real = di_uv_use.real
        di_uv_img = di_uv_use.imag
        # Use rebin to resize, apply to real and complex separately
        di_uv_real = rebin(di_uv_real, (elements, dimension))
        di_uv_img = rebin(di_uv_img, (elements, dimension))
        # Combine real and complex back together
        di_uv_use = di_uv_real + di_uv_img * 1j

    # Apply padding if it was supplied
    if pad_uv_image is not None:
        # dimension_new = int(np.max([np.max([dimension, elements]) * pad_uv_image, np.max([dimension, elements])]))
        # di_uv1 = np.zeros((dimension_new, dimension_new), dtype = "complex")
        # di_uv1[dimension_new // 2 - elements // 2 : dimension_new // 2 + elements // 2,
        #        dimension_new // 2 - dimension // 2 : dimension_new // 2 + dimension // 2] = di_uv_use
        di_uv1 = np.pad(
            di_uv_use, (np.max([dimension, elements]) // 2) * (int(pad_uv_image) - 1)
        )
        di_uv_use = di_uv1 * (pad_uv_image**2)

    # FFT normalization
    if degpix is not None:
        di_uv_use /= np.radians(degpix) ** 2

    # Multivariate Fast Fourier Transform
    dirty_image = np.fft.fftshift(
        np.fft.fftn(np.fft.fftshift(di_uv_use), norm="forward")
    )
    if not not_real:
        dirty_image = dirty_image.real

    # filter_uv_optimal produces images that are weighted by one factor of the beam
    # Weight by an additional factor of the beam to align with FHD's convention
    if pyfhd_config["image_filter"] == "filter_uv_optimal" and beam_ptr is not None:
        dirty_image *= beam_ptr

    # If we are returning complex, make sure its complex
    if not_real:
        dirty_image = dirty_image.astype(np.complex128)
    else:
        dirty_image = dirty_image.real
    # Normalize by the matrix given, if it was given
    if normalization is not None:
        dirty_image *= normalization

    # Return
    return dirty_image, filter, normalization


def grid_beam_per_baseline(
    psf: dict,
    pyfhd_config: dict,
    logger: Logger,
    uu: NDArray[np.float64],
    vv: NDArray[np.float64],
    ww: NDArray[np.float64],
    l_mode: NDArray[np.float64],
    m_mode: NDArray[np.float64],
    n_tracked: NDArray[np.float64],
    frequency_array: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xmin_use: int,
    ymin_use: int,
    freq_i: NDArray[np.integer],
    bt_index: NDArray[np.integer],
    polarization: int,
    image_bot: int,
    image_top: int,
    psf_dim3: int,
    box_matrix: NDArray[np.complex128],
    vis_n: int,
    beam_int: NDArray[np.complex128] | None = None,
    beam2_int: NDArray[np.complex128] | None = None,
    n_grp_use: NDArray[np.integer] | None = None,
    degrid_flag: bool = False,
):
    """
    Calculate the contribution of each baseline to the static {u,v} grid using the corrective phases in
    image-space, as detailed in J. Line's thesis "PUMA and MAJICK: cross-matching and imaging techniques
    for a detection of the epoch of reionisation"

    Parameters
    ----------
    psf : dict
        Beam metadata dictionary
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger
    uu : NDArray[np.float64]
        1D array of the u-coordinate of selected baselines in light travel time
    vv : NDArray[np.float64]
        1D array of the v-coordinate of selected baselines in light travel time
    ww : NDArray[np.float64]
        1D array of the w-coordinate of selected baselines in light travel time
    l_mode : NDArray[np.float64]
        Directional-cosine l of pixel centers of the hyperresolved beam
    m_mode : NDArray[np.float64]
        Directional-cosine m of pixel centers of the hyperresolved beam
    n_tracked : NDArray[np.float64]
        Directional-cosine n of pixel centers of the phase-tracked hyperresolved beam
    frequency_array : NDArray[np.float64]
        Array of selected frequencies in Hz
    x : NDArray[np.float64]
        1D array of gridding extent and resolution in the x-direction in wavelengths
    y : NDArray[np.float64]
        1D array of gridding extent and resolution in the y-direction in wavelengths
    xmin_use : int
        The minimum x-pixel that each selected baseline contributes to
    ymin_use : int
        The minimum y-pixel that each selected baseline contributes to
    freq_i : NDArray[np.integer]
        The current frequency index
    bt_index : NDArray[np.integer]
        The current baseline/time index
    polarization : int
        The current polarization index
    image_bot : int
       The bottom-most pixel index that the image-space hyperresolved beam contributes to
    image_top : int
        The top-most pixel index that the image-space hyperresolved beam contributes to
    psf_dim3 : int
        The pixel area of the psf footprint on the {u,v} grid
    box_matrix : NDArray[np.complex128]
        A 2D array of the number of visibilities to grid and the area of each visibility on the static {u,v} grid
    vis_n : int
        The number of visibilities to grid
    beam_int : NDArray[np.complex128] | None, optional
        The integral of the beam sensitivity in {u,v} space, by default None
    beam2_int : NDArray[np.complex128] | None, optional
        The integral of the squared beam sensitivity in {u,v} space, by default None
    n_grp_use : NDArray[np.integer] | None, optional
        The number of baselines in the current grouping, by default None
    degrid_flag : bool, optional
        Perform degridding instead of gridding, by default False

    Returns
    -------
    box_matrix : NDArray[np.complex128]
        The kernel values on the static {u,v} grid for each visibility
    """

    psf_dim = psf["dim"]
    psf_resolution = psf["resolution"]
    if isinstance(psf, h5py.File):
        psf_dim = psf_dim[0]
        psf_resolution = psf_resolution[0]
    # Loop over all visibilities that fall within the chosen visibility box
    for ii in range(vis_n):
        # Pixel center offset phases
        deltau_l = l_mode * (
            uu[bt_index[ii]] * frequency_array[freq_i[ii]] - x[xmin_use + psf_dim // 2]
        )
        deltav_m = m_mode * (
            vv[bt_index[ii]] * frequency_array[freq_i[ii]] - y[ymin_use + psf_dim // 2]
        )
        # w term offset phase
        w_n_tracked = n_tracked * ww[bt_index[ii]] * frequency_array[freq_i[ii]]

        # Generate a UV beam from the image space beam, offset by calculated phases
        psf_base_superres, _, _ = dirty_image_generate(
            psf["image_info"]["image_power_beam_arr"][polarization]
            * np.exp(2 * pi * (0 + 1j) * (-w_n_tracked + deltau_l + deltav_m)),
            pyfhd_config,
            logger,
            not_real=True,
        )
        psf_base_superres = psf_base_superres[
            image_bot : image_top + 1, image_bot : image_top + 1
        ]

        # A quick way to sum down the image by a factor of 2 in both dimensions.
        # A 4x4 example where we sum down by a factor of 2
        #
        # 1  2  3  4           1  2           1  2           1  2  5  6            14 46           14 22
        # 5  6  7  8    -->    3  4    -->    5  6    -->    9  10 13 14    -->    22 54    -->    46 54
        # 9  10 11 12                         9  10
        # 13 14 15 16          5  6           13 14          3  4  7  8
        #                      7  8                          11 12 15 16
        #                                     3  4
        #                      9  10          7  8
        #                      11 12          11 12
        #                                     15 16
        #                      13 14
        #                      15 16
        d = psf_base_superres.shape
        # Note columns and rows are swapped from IDL so nx is now rows!
        nx = d[0] // psf_resolution
        ny = d[1] // psf_resolution
        # The same result of IDL in numpy is np.reshape, with shape swapping rows and columns, then doing transpose of this shape
        psf_base_superres = np.reshape(
            psf_base_superres, [psf_resolution * ny, nx, psf_resolution]
        )
        psf_base_superres = np.transpose(psf_base_superres, [1, 0, 2])
        psf_base_superres = np.reshape(psf_base_superres, [ny, nx, psf_resolution**2])
        psf_base_superres = np.sum(psf_base_superres, -1)
        psf_base_superres = np.transpose(psf_base_superres)
        psf_base_superres = np.reshape(psf_base_superres, psf_dim**2)
        start = psf_dim3 * ii
        end = start + psf_base_superres.size
        box_matrix_iter = box_matrix.flat
        box_matrix_iter[start:end] = psf_base_superres

    # Subtract off a small clip, set negative indices to 0, and renomalize.
    # This is a modification of the look-up-table beam using a few assumptions
    # to make it faster/feasible to run.
    # Modifications: done per group of baselines that fit within the current box,
    # rather than individually. region_grow is not used to find a contiguous
    # edge around the beam to cut because it is too slow.
    if pyfhd_config["beam_clip_floor"]:
        psf_val_ref = np.sum(box_matrix, 1)
        psf_amp = np.abs(box_matrix)
        psf_mask_threshold_use = np.max(psf_amp) / psf["beam_mask_threshold"]
        psf_amp -= psf_mask_threshold_use
        psf_phase = np.arctan2(box_matrix.imag, box_matrix.real)
        psf_amp = np.maximum(psf_amp, np.zeros_like(psf_amp))
        box_matrix = psf_amp * np.cos(psf_phase) + (0 + 1j) * psf_amp * np.sin(
            psf_phase
        )
        ref_temp = np.sum(box_matrix, -1)
        box_matrix[:vis_n, :] *= np.reshape(
            psf_val_ref / ref_temp, (psf_val_ref.size, 1)
        )

    if (
        degrid_flag
        and beam_int is not None
        and beam2_int is not None
        and n_grp_use is not None
    ):
        # Calculate the beam and beam^2 integral (degridding)
        beam_int_temp = np.sum(box_matrix, 0) / psf_resolution**2
        beam2_int_temp = np.sum(np.abs(box_matrix) ** 2, 0) / psf_resolution**2
        for ii in range(np.size(freq_i)):
            beam_int[freq_i[ii]] += beam_int_temp[ii]
            beam2_int[freq_i[ii]] += beam2_int_temp[ii]
            n_grp_use[freq_i[ii]] += 1

    return box_matrix


def visibility_count(
    obs: dict,
    psf: dict,
    params: dict,
    vis_weights: NDArray[np.float64],
    logger: Logger,
    fi_use: NDArray[np.integer] | None = None,
    bi_use: NDArray[np.integer] | None = None,
    mask_mirror_indices: bool = False,
    no_conjugate: bool = False,
    fill_model_visibilities: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate the number of contributing visibilities per pixel on the static {u,v} grid

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    params : dict
        Visibility metadata dictionary
    vis_weights : NDArray[np.float64]
        Weights (flags) of the visibilities
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options set for a pyfhd run
    logger : Logger
        pyfhd's logger
    fi_use : NDArray[np.integer] | None, optional
        Frequency index array for gridding, i.e. gridding all frequencies for continuum images, by default None
    bi_use : NDArray[np.integer] | None, optional
        Baseline index array for gridding, i.e even vs odd time stamps, by default None
    mask_mirror_indices : bool, optional
        Exclude baselines mirrored along the v-axis, by default False
    no_conjugate : bool, optional
        Do not perform the conjugate mirror of the {u,v} plane, by default False
    fill_model_visibilities : bool, optional
        Calculate baseline grid locations for all baselines, regardless of flags, by default False

    Returns
    -------
    uniform_filter : NDArray[np.float64]
        2D array of number of contributing visibilities per pixel on the {u,v} grid
    """

    # Retrieve info from the data structures
    dimension = int(obs["dimension"])
    elements = int(obs["elements"])
    psf_dim = psf["dim"]
    if isinstance(psf, h5py.File):
        psf_dim = psf_dim[0]

    baselines_dict = baseline_grid_locations(
        obs,
        psf,
        params,
        vis_weights,
        logger,
        bi_use=bi_use,
        fi_use=fi_use,
        mask_mirror_indices=mask_mirror_indices,
        fill_model_visibilities=fill_model_visibilities,
    )
    # Retrieve the data we need from baselines_dict
    bin_n = baselines_dict["bin_n"]
    bin_i = baselines_dict["bin_i"]
    xmin = baselines_dict["xmin"]
    ymin = baselines_dict["ymin"]
    ri = baselines_dict["ri"]
    n_bin_use = baselines_dict["n_bin_use"]
    # Remove baselines_dict
    del baselines_dict

    uniform_filter = np.zeros((elements, dimension))
    # Get the flat iterations of the xmin and ymin arrays.
    xmin_iter = xmin.flat
    ymin_iter = ymin.flat
    for bi in range(n_bin_use):
        idx = ri[ri[bin_i[bi]]]
        # Should all be the same, but don't want an array
        xmin_use = xmin_iter[idx]
        ymin_use = ymin_iter[idx]
        # Please note that due to precision differences, the result will be different compared to IDL FHD
        uniform_filter[
            xmin_use : xmin_use + psf_dim, ymin_use : ymin_use + psf_dim
        ] += bin_n[bin_i[bi]]

    if not no_conjugate:
        uniform_filter = (uniform_filter + conjugate_mirror(uniform_filter)) / 2

    return uniform_filter


def holo_mapfn_convert(map_fn, psf_dim, dimension, elements=None, norm=1, threshold=0):
    """
    Convert pointer array holographic map function to a sparse matrix. May need to be deprecated.
    The mapping functions were not translated into `visibility_grid` at the time of translation as
    it wasn't clear what pyfhd was going to use for sparse/large arrays at the time. If you wish to
    implement the mapping functions, I suggest using HDF5 chunk loading for the mapping function.

    Parameters
    ----------
    map_fn: np.ndarray
        Pointer array holographic map function
    psf_dim: int, float
        The number of pixels in one direction of the psf {u,v} footprint
    dimension: int
        The number of pixels in the u-direction
    elements: None, optional
        The number of pixels in the v-direction
    norm: int
        The normalization of the holographic mapping function (i.e. the number of visibilities)
    threshold: int, float, optional
        Include values from the holographic map function which are above a provided threshold

    Returns
    -------
    map_fn: np.recarray
        Sparse array holographic map function
    """
    # Set up all the necessary arrays and numbers
    if elements is None:
        elements = dimension
    psf_dim2 = 2 * psf_dim
    sub_xv = meshgrid(psf_dim2, 1) - psf_dim
    sub_yv = meshgrid(psf_dim2, 2) - psf_dim
    # Generate an array of shape elements x dimension
    n_arr = np.zeros((elements, dimension))
    # Replace the values of n_arr with the number from each array in mapfn that exceeds the threshold
    for xi in range(dimension - 1):
        for yi in range(elements - 1):
            temp_arr = map_fn[xi, yi]
            n1 = np.size(np.where(abs(temp_arr) > threshold))
            n_arr[xi, yi] = n1
    # Get the ones to use
    i_use = np.where(n_arr)
    # Get the amount we're using
    i_use_size = np.size(i_use)
    # If we aren't using any then return 0
    if i_use_size == 0:
        return 0

    # Get the reverse indices
    _, _, ri = histogram(i_use, min=0)
    # Create zeros of the same size as what we're using
    sa = np.zeros(i_use_size)
    ija = np.zeros(i_use_size)
    # Fill in the sa and ija arrays
    for index in range(i_use_size):
        i = i_use[index]
        xi = i % dimension
        yi = np.floor(i / dimension)
        map_fn_sub = map_fn[xi, yi]
        j_use = np.where(np.abs(map_fn_sub) > threshold)

        xii_arr = sub_xv[j_use] + xi
        yii_arr = sub_yv[j_use] + yi
        sa[index] = map_fn_sub[j_use]
        ija[index] = ri[ri[xii_arr + dimension * yii_arr]]
    # Return as record array
    mapfn = np.recarray(
        ija,
        sa,
        i_use,
        norm,
        1,
        dtype=[
            (("ija", "IJA"), "int"),
            (("sa", "SA"), "int"),
            (("i_use", "I_USE"), "int"),
            (("norm", "NORM"), "float"),
            (("indexed", "INDEXED"), ">i2"),
        ],
    )
    return mapfn


def crosspol_reformat(image_uv: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Reformat the cross-polarizations (i.e. XY and YX) as pseudo Stokes Q and U. This helps to
    avoid complex numbers in creating images -- however, this is an imperfect assumption.

    Parameters
    ----------
    image_uv : NDArray[np.complex128]
        A 2D {u,v} plane in four linear polarizations.

    Returns
    -------
    image_uv : NDArray[np.complex128]
        A 2D {u,v} plane in two linear polarizations, pseudo Stokes Q, and pseudo Stokes U.
    """
    # instrumental -> Stokes
    # Since inverse keyword in FHD isn't used or explained
    # anywhere else, if you want it, add it as an option to
    # the pyfhd_config with some help text
    crosspol_image = (image_uv[2] - conjugate_mirror(image_uv[3])) / 2
    image_uv[2] = crosspol_image
    image_uv[3] = conjugate_mirror(crosspol_image)
    return image_uv

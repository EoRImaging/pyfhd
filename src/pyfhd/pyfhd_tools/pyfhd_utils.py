import subprocess
from copy import deepcopy
from logging import Logger
from math import factorial, pi
from sys import exit
import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import label, median_filter


@njit
def get_bins(
    min: int | float, max: int | float, bin_size: int
) -> NDArray[np.float64 | np.int64]:
    """
    Calculates the bins for the histogram and reverse indices based on a
    minimum and maximum value, plus a given bin size. It mirrors IDL's way of
    calculating the bin edges. In IDL bins seem to be right hand side of the bin open
    like IDL, even at the end. However, in comparison to NumPy, the last bin is always
    the maximum value. This does utilize numba and parallelization to make it faster.

    Parameters
    ----------
    min : int | float
        The minimum chosen for the histogram
    max : int | float
        The maximum chosen for the histogram
    bin_size : int
        The bin size chosen for the histogram. This histogram always uses bins
        of equal widths.

    Returns
    -------
    bins: NDArray[np.float64 | np.int64]
        A NumPy array of all the bins ranging from min (bin[0]) to the max (bin[-1]). The step
        is bin_size.

    See Also
    --------
    histogram: Calculates the bins, histogram and reverse indices
    get_hist: Calculates histogram only
    get_ri: Calculates the reverse indices only
    """

    return np.arange(min, max + bin_size, bin_size)


@njit
def get_hist(
    data: NDArray[np.floating | np.integer | np.complexfloating],
    bins: NDArray[np.float64 | np.int64],
    min: int | float,
    max: int | float,
) -> NDArray[np.int64]:
    """
    Calculates the histogram based on the given bins and data, taking into account
    the minimum and maximum

    Parameters
    ----------
    data : NDArray[np.floating | np.integer | np.complexfloating]
        A NumPy array that is of one dtype float, int, or complex. Cannot be object
    bins : NDArray[np.float64 | np.int64]
        A NumPy array of bins for the histogram
    min : int | float
        The minimum set for the data
    max : int | float
        The maximum set for the data

    Returns
    -------
    hist: NDArray[np.int64]
        A histogram of the data using the bins found in the bins array.

    See Also
    --------
    histogram: Calculates the bins, histogram and reverse indices.
    get_bins: Calculates the bins only
    get_ri: Calculates the reverse indices only
    """

    bin_l = bins.size
    # Setup the histogram
    hist = np.zeros(bin_l, dtype=np.int64)
    # Setup the things required for the indexing
    n = bin_l - 1
    bin_min = bins[0]
    bin_max = bins[-1]
    # In the case that the bin min and bin max is the same,
    # then finding the index doesn't matter so set the divider
    # to 1 (as there is only 1 bin!)
    if bin_min == bin_max:
        bin_divider = 1
    else:
        bin_divider = bin_max - bin_min
    # Now loop through the data
    for idx in range(data.size):
        # Check if its inside the range we set
        if data[idx] < min or data[idx] > max:
            continue
        # Calculate the index for indices and histogram
        bin_i = int(n * (data[idx] - bin_min) / bin_divider)
        # Add to the histogram
        hist[bin_i] += 1
    return hist


@njit
def get_ri(
    data: NDArray[np.floating | np.integer | np.complexfloating],
    bins: NDArray[np.float64 | np.int64],
    hist: NDArray[np.int64],
    min: int | float,
    max: int | float,
) -> NDArray[np.int64]:
    """
    Calculates the reverse indices of a data and histogram.
    The function replicates IDL's REVERSE_INDICES keyword within
    their HISTOGRAM function. The reverse indices array which is returned
    by this function can be hard to understand at first, I will explain it here
    and also link JD Smith's famous article on IDL's HISTOGRAM.

    The reverse indices array is two vectors concatenated together. The first vector contains
    indexes for the second vector, this vector should be the size of bins + 1.
    The second vector contains indexes from the data itself, and should be the size of data.
    The justification for having such an array is to quickly make adjustments to certain bins
    without having to search the array multiple times, thus avoiding multiple O(data.size) loops.

    The first vector indexes contain the starting positions of each bin in the second vector. For example,
    between the indexes given by first_vector[0] and first_vector[1] of the second vector should be all the
    indexes of bins[0] from inside the data. So if I wanted to make adjustments to the entire first bin,
    and only the first bin I can use the reverse indices array, ri to do this. Let's say I wanted to flag
    all values of bins[0] with -1 for some reason to make them invalid in other calculations with the data,
    then I could do this:

    `data[ri[ri[0] : ri[1]]] = -1`

    Or more generally

    `data[ri[ri[i] : ri[i + 1]]] = -1`

    Where i is 0 <= i <= bins.size.

    If you wish to gain a better understanding of how this get_ri function works, and the associated
    histogram function I have created here, please use the link given in the Notes section. This
    link will take you JD Smith's article on IDL's HISTOGRAM, it is an article which explains the
    IDL HISTOGRAM function better than IDL's own documentation. If you must gain a deeper understanding,
    read it once, gasp and get your shocks and many cries of why out of your system, then read it again.
    And keep reading it till you understand, as per the editor's note on the article:

    "...If you read it enough, the secrets of the command will be revealed to you. Stranger things have happened"

    Parameters
    ----------
    data : NDArray[np.floating | np.integer | np.complexfloating]
        A NumPy array of the data
    bins : NDArray[np.float64 | np.int64]
        A NumPy array containing the bins for the histogram
    hist : NDArray[np.int64]
        A NumPy array containing the histogram
    min : int | float
        The minimum for the dataset
    max : int | float
        The maximum for the dataset

    Returns
    -------
    ri : NDArray[np.int64]
        An array containing the reverse indices, which is two vectors, the first vector containing indexes for the second vector.
        The second vector contains indexes for the data.

    See Also
    ---------
    histogram: Calculates the bins, histogram and reverse indices.
    get_bins: Calculates the bins only
    get_hist: Calculates histogram only

    Notes
    ------
    'HISTOGRAM: The Breathless Horror and Disgust' : http://www.idlcoyote.com/tips/histogram_tutorial.html
    """

    bin_l = bins.size
    # Setup the first vector
    data_idx = bin_l + 1
    first_v = [bin_l + 1]
    # Add the bins
    for bin in hist:
        data_idx += bin
        first_v.append(data_idx)
    # Setup the reverse indices
    ri = np.zeros(bin_l + 1 + data.size, dtype=np.int64)
    ri[0 : bin_l + 1] = first_v
    # Create a tracking array to keep track of where we are in the data indexing
    tracker = np.array(first_v)
    # Setup the things required for the indexing
    n = bin_l - 1
    bin_min = bins[0]
    bin_max = bins[-1]
    if bin_min == bin_max:
        bin_divider = 1
    else:
        bin_divider = bin_max - bin_min
    # Setup counter to remove elements in case of min or max being set
    counter = 0
    for idx in range(data.size):
        # Check if its inside the range we set
        if data[idx] < min or data[idx] > max:
            counter += 1
            continue
        # Calculate the index for indices and histogram
        bin_i = int(n * (data[idx] - bin_min) / bin_divider)
        # Record the current index and update the tracking array
        ri[tracker[bin_i]] = idx
        tracker[bin_i] += 1
    ri = ri[: ri.size - counter]
    return ri


def histogram(
    data: NDArray[np.floating | np.integer | np.complexfloating],
    bin_size: int = 1,
    num_bins: int | None = None,
    min: int | float | None = None,
    max: int | float | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64 | np.int64], NDArray[np.int64]]:
    """
    The histogram function combines the use of the get_bins, get_hist and get_ri
    functions into one function. For the descriptions and docs of those functions
    look in See Also. This function will return the histogram, bin/bin_edges and
    the reverse indices.

    Parameters
    ----------
    data : NDArray[np.floating | np.integer | np.complexfloating]
        A NumPy array containing the data we want a histogram of
    bin_size : int, optional
        Sets the bin size for this histogram, by default 1
    num_bins : int | None, optional
        Set the number of bins this does override bin_size completely, by default None
    min :  int | float | None, optional
        Set a minimum for the dataset, by default None
    max :  int | float | None, optional
        Set a maximum for the dataset, by default None

    Returns
    -------
    hist : NDArray[np.int64]
        The histogram of the data
    bins : NDArray[np.float64 | np.int64]
        The bins of the histogram
    ri : NDArray[np.int64]
        The reverse indices array for the histogram and data

    See Also
    --------
    get_bins: Calculates the bins only
    get_hist: Calculates histogram only
    get_ri: Calculates the reverse indices only
    """

    # If the minimum has not been set, set it
    if min is None:
        min = np.min(data)
    # If the maximum has not been set, set it
    # Check if the max argument was used, set to True, if we set max here by data turn it off.
    if max is None:
        max = np.max(data)
    # If the number of bins has been set use that
    if num_bins is not None:
        bin_size = (max - min) / num_bins
    # Need to add checks if max is below min or max below min
    if min > max:
        max = min
    if max < min:
        min = max
    # IDL uses the bin_size as equal throughout min to max
    bins = get_bins(min, max, bin_size)
    # However, if we set a max, we must adjust the last bin to max according to IDL specifications
    # And we only do this in the case max was by an argument
    if (bins[-1] > max) or num_bins is not None:
        bins = bins[:-1]
    # Flatten the data
    data_flat = data.flatten()
    # Get the histogram
    hist = get_hist(data_flat, bins, min, max)
    # Get the reverse indices
    ri = get_ri(data_flat, bins, hist, min, max)
    # Return
    return hist, bins, ri


def l_m_n(
    obs: dict,
    psf: dict | h5py.File,
    obsdec: float | None = None,
    obsra: float | None = None,
    declination_arr: NDArray[np.floating] | None = None,
    right_ascension_arr: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the directional cosines l,m,n given an RA, Dec, and phase centre for
    each pixel in the input arrays given, or from the hyperresolved RA/Dec array
    created during beam creation.

    Parameters
    ----------
    obs: dict
        Observation metadata dictionary.
    psf: dict | h5py.File
        Beam dictionary.
    obsdec: float | None, optional
        The Dec phase centre for the observation, by default None and set to
        the phase centre saved in the observation metadata dicitonary
        (obs['obsdec'])
    obsra: float | None, optional
        The RA phase centre for the observation, by default None and set to
        the phase centre saved in the observation metadata dicitonary
        (obs['obsra'])
    declination_arr: NDArray[np.floating] | None, optional
        An array of declinations, by default None and set to the hyperresolved Dec
        array in psf['image_info']['dec_arr']
    right_ascension_arr: NDArray[np.floating] | None, optional
        An array of right ascensions, by default None and set to the hyperresolved RA
        array in psf['image_info']['ra_arr']

    Returns
    -------
    l_mode : NDArray[np.float64]
        Directional cosine l array, or the cosine of the vector that contributes
        to the x-axis
    m_mode : NDArray[np.float64]
        Directional cosine m array, or the cosine of the vector that contributes
        to the y-axis
    n_tracked : NDArray[np.float64]
        Directional cosine n array, or the cosine of the vector that contributes
        to the z-axis, dependent on the phase centre.
    """
    # If the variables passed through are None them
    if obsdec is None:
        obsdec = obs["obsdec"]
    if obsra is None:
        obsra = obs["obsra"]
    if declination_arr is None:
        declination_arr = psf["image_info"]["dec_arr"]
    if right_ascension_arr is None:
        right_ascension_arr = psf["image_info"]["ra_arr"]

    # Convert all the degrees given into radians
    obsdec = np.radians(obsdec)
    obsra = np.radians(obsra)
    declination_arr = np.radians(declination_arr)
    right_ascension_arr = np.radians(right_ascension_arr)

    # Calculate l mode, m mode and the phase-tracked n mode of pixel centers
    cdec0 = np.cos(obsdec)
    sdec0 = np.sin(obsdec)
    cdec = np.cos(declination_arr)
    sdec = np.sin(declination_arr)
    cdra = np.cos(right_ascension_arr - obsra)
    sdra = np.sin(right_ascension_arr - obsra)
    l_mode = sdra * cdec
    m_mode = cdec0 * sdec - cdec * sdec0 * cdra
    # n=1 at phase center, so reference from there for phase tracking
    n_tracked = (sdec * sdec0 + cdec * cdec0 * cdra) - 1

    # find any NaN values
    nan_vals = np.where(np.isnan(n_tracked))
    # If any found, replace them with 0's
    if np.size(nan_vals) > 0:
        n_tracked[nan_vals] = 0
        l_mode[nan_vals] = 0
        m_mode[nan_vals] = 0

    # Return the modes
    return l_mode, m_mode, n_tracked


def rebin_columns(
    a: NDArray[np.integer | np.floating | np.complexfloating],
    ax: int,
    shape: tuple,
    col_sizer: int,
) -> NDArray[np.integer | np.floating | np.complexfloating]:
    """
    Performs expansion on the columns of a 1D or 2D array using interpolation to fill in the values that are created
    by expanding in the space between existing values. This function assumes the rows have already been expanded
    to the required number.

    Parameters
    ----------
    a : NDArray[np.integer  |  np.floating  |  np.complexfloating]
        An array to be expanded
    ax : int
        The axis we're expanding, almost always set to 1
    shape : tuple
        The shape of the original array
    col_sizer : int
        The number of columns we're adding in

    Returns
    -------
    rebinned : NDArray[np.integer | np.floating | np.complexfloating]
        An interpolated array of a containing shape[1] * col_sizer columns

    See Also
    --------
    pyfhd.pyfhd_tools.pyfhd_utils.rebin_rows : Expand the number of rows through interpolation
    pyfhd.pyfhd_tools.pyfhd_utils.rebin : Expand or Contract an array based on a given shape
    """
    # tile the range of col_sizer
    tiles = np.tile(np.arange(col_sizer), (shape[0], shape[1] // col_sizer - 1))
    # Get the differences between the columns
    differences = np.diff(a, axis=ax) / col_sizer
    # Multiply this by the tiles
    inferences_non_pad = np.repeat(differences, col_sizer, axis=ax) * tiles
    # Pad the zeros for the last two rows, and remove the extra zeros to make inferences same shape as desired shape
    inferences = np.pad(inferences_non_pad, (0, col_sizer))[:-col_sizer]
    if np.issubdtype(a.dtype, np.integer):
        inferences = np.floor(inferences).astype("int")
    # Now get our final array by adding the repeat of our rows rebinned to the inferences
    rebinned = inferences + np.repeat(a, col_sizer, axis=ax)
    return rebinned


def rebin_rows(
    a: NDArray[np.integer | np.floating | np.complexfloating],
    ax: int,
    shape: tuple,
    old_shape: tuple,
    row_sizer: int,
) -> NDArray[np.integer | np.floating | np.complexfloating]:
    """
    Performs expansion on the rows of array `a` to the number of rows in shape[0] using interpolation to fill between any
    new values added when adding new rows between existing values.

    Parameters
    ----------
    a : NDArray[np.integer  |  np.floating  |  np.complexfloating]
        The array to be rebinned
    ax : int
        The axis we're expanding, almost always set to 0
    shape : tuple
        The new shape we're expanding to
    old_shape : tuple
        The shape of the array a
    row_sizer : int
        The number of rows we're expanding by

    Returns
    -------
    row_rebinned : NDArray[np.integer | np.floating | np.complexfloating]
        The interpolated array with row_sizer extra columns between existing rows

    See Also
    --------
    pyfhd.pyfhd_tools.pyfhd_utils.rebin_columns : Expand the number of columns through interpolation
    pyfhd.pyfhd_tools.pyfhd_utils.rebin : Expand or Contract an array based on a given shape
    """
    # Tile the range of row_sizer
    tiles = np.tile(
        np.array_split(np.arange(row_sizer), row_sizer),
        ((shape[0] - row_sizer) // row_sizer, old_shape[1]),
    )
    # Get the differences between values
    differences = np.diff(a, axis=ax) / row_sizer
    # Multiply differences array by tiles to get desired bins
    inferences_non_pad = np.repeat(differences, row_sizer, axis=ax) * tiles
    # Pad the inferences to get the same shape as above
    inferences = np.pad(inferences_non_pad, (0, row_sizer))[:, :-row_sizer]
    if np.issubdtype(a.dtype, np.integer):
        inferences = np.floor(inferences).astype("int")
    # Add this to the original array that has been repeated to match the size of inference
    row_rebinned = inferences + np.repeat(a, row_sizer, axis=ax)
    return row_rebinned


def rebin(
    a: NDArray[np.integer | np.floating | np.complexfloating],
    shape: ArrayLike,
    sample: bool = False,
) -> NDArray[np.integer | np.floating | np.complexfloating]:
    """
    Resizes a 2D array by averaging or repeating elements, new dimensions must be integral factors of original dimensions.

    In the case of expanding an existing array, rebin will interpolate between the original values with a linear function.
    In the case of compressing an existing array, rebin will average

    Parameters
    ----------
    a : NDArray[np.integer | np.floating | np.complexfloating]
        Input array.
    new_shape : ArrayLike
        Shape of the output array in (rows, columns)
        Must be a factor or multiple of a.shape
    sample: bool, optional
        Use this to get samples using rebin, rather than interpolation, by default False.

    Returns
    -------
    rebinned : NDArray[np.integer | np.floating | np.complexfloating]
        If the new shape is smaller of the input array, the data are averaged,
        if the new shape is bigger array elements are repeated and interpolated

    Examples
    --------
    >>> test = np.array([0,10,20,30])
    >>> rebin(test, (1,8)) # Expand Columns
    array([ 0,  5, 10, 15, 20, 25, 30, 30])
    >>> rebin(test, (2,8)) # Expand Rows and Columns
    array([[ 0,  5, 10, 15, 20, 25, 30, 30],
           [ 0,  5, 10, 15, 20, 25, 30, 30]])
    >>> data = np.array([[ -5,   4,   2,  -8,   1],
                         [  3,   0,   5,  -5,   1],
                         [  6,  -7,   4,  -4,  -8],
                         [ -1,  -5, -14,   2,   1]])
    >>> rebin(data, (8,10)) # 2D Array example
    array([[ -5,  -1,   4,   3,   2,  -3,  -8,  -4,   1,   1],
           [ -1,   0,   2,   2,   3,  -2,  -7,  -3,   1,   1],
           [  3,   1,   0,   2,   5,   0,  -5,  -2,   1,   1],
           [  4,   0,  -4,   0,   4,  -1,  -5,  -5,  -4,  -4],
           [  6,  -1,  -7,  -2,   4,   0,  -4,  -6,  -8,  -8],
           [  2,  -2,  -6,  -6,  -5,  -3,  -1,  -3,  -4,  -4],
           [ -1,  -3,  -5, -10, -14,  -6,   2,   1,   1,   1],
           [ -1,  -3,  -5, -10, -14,  -6,   2,   1,   1,   1]])
    >>> rebin(data, (2,5)) # Compression
    array([[-1,  2,  3, -6,  1],
           [ 2, -6, -5, -1, -3]])
    >>> to_compress = np.array([[3, 9, 7, 0, 1, 5],
                                [7, 7, 1, 9, 7, 3],
                                [9, 2, 2, 3, 1, 1],
                                [0, 3, 5, 0, 4, 3],
                                [5, 7, 7, 1, 9, 1],
                                [7, 2, 1, 1, 3, 0]])
    >>> rebin(to_compress, (3,3)) # Compression
    array([[6, 4, 4],
           [3, 2, 2],
           [5, 2, 3]])

    References
    ----------
    [1] https://stackoverflow.com/a/8090605

    See Also
    --------
    pyfhd.pyfhd_tools.pyfhd_utils.rebin_rows : Expand the number of rows through interpolation
    pyfhd.pyfhd_tools.pyfhd_utils.rebin_columns : Expand the number of columns through interpolation
    """
    old_shape = a.shape
    # Prevent more processing than needed if we want the same shape
    if old_shape == shape:
        return a
    if len(old_shape) == 1:
        old_shape = (1, old_shape[0])
    # Sample the original array using rebin
    if sample:
        # If its a 1D array then...
        if old_shape[0] == 1:
            if shape[1] > old_shape[1]:
                rebinned = np.repeat(a, shape[1] // old_shape[1], axis=0)
            else:
                rebinned = a[:: old_shape[1] // shape[1]]
        # Assume its a 2D array
        else:
            # Do the Rows first
            if shape[0] >= old_shape[0]:
                # Expand Rows
                rebinned = np.repeat(a, shape[0] // old_shape[0], axis=0)
            else:
                # Compress Rows
                rebinned = a[:: old_shape[0] // shape[0], :]
            # Then do the columns
            if shape[1] >= old_shape[1]:
                # Expand Columns
                rebinned = np.repeat(rebinned, shape[1] // old_shape[1], axis=1)
            else:
                # Compress columns
                rebinned = rebinned[:, :: old_shape[1] // shape[1]]
        # Return the rebinned without adjusting dtype as none of the functions above change it
        return rebinned

    # If we are downsizing
    elif shape[0] < old_shape[0] or shape[1] < old_shape[1]:
        if (max(old_shape[0], shape[0]) % min(old_shape[0], shape[0]) != 0) or (
            max(old_shape[1], shape[1]) % min(old_shape[1], shape[1]) != 0
        ):
            raise ValueError("Your new shape should be a factor of the original shape")
        # If we are increasing the rows or columns and reducing the other, increase them now and change the old shape
        if shape[0] > old_shape[0]:
            a = np.tile(a, (shape[0], 1))
            old_shape = a.shape
        elif shape[1] > old_shape[1]:
            a = np.tile(a, (1, shape[1]))
            old_shape = a.shape
        # Create the shape we need (rows, rows that can fit in old_shape, cols, cols that can fit into old_shape)
        sh = shape[0], old_shape[0] // shape[0], shape[1], old_shape[1] // shape[1]
        # Create the 4D array
        rebinned = np.reshape(a, sh)
        # Get the average of the columns first
        rebinned = rebinned.mean(-1)
        # To ensure that we get the result same as IDL
        # it seems to fix the values after every calculation if integer
        if np.issubdtype(a.dtype, np.integer):
            rebinned = np.fix(rebinned).astype("int")
        # Now get it for the rows
        rebinned = rebinned.mean(1)
        # If we are expecting 1D array ensure it gets returned as a 1D array
        if shape[0] == 1:
            rebinned = rebinned[0]
        # To ensure that we get the result same as IDL
        # it seems to fix the values after every calculation if integer
        if np.issubdtype(a.dtype, np.integer):
            rebinned = np.fix(rebinned).astype("int")

    # Otherwise we are expanding
    else:
        if shape[0] % old_shape[0] != 0 or shape[1] % old_shape[1] != 0:
            raise ValueError(
                "Your new shape should be a multiple of the original shape"
            )
        # Get the size changes of the row and column separately
        row_sizer = shape[0] // old_shape[0]
        col_sizer = shape[1] // old_shape[1]
        ax = 0
        # If 1D array then do along the columns
        if old_shape[0] == 1:
            rebinned = rebin_columns(a, ax, shape, col_sizer)
            if shape[0] == 1:
                rebinned = rebinned[0]
        # Else its a 2D array
        else:
            row_rebinned = rebin_rows(a, ax, shape, old_shape, row_sizer)
            # If it matches the new shape, then return it
            if row_rebinned.shape == shape:
                return row_rebinned
            else:
                ax = 1
                rebinned = rebin_columns(row_rebinned, ax, shape, col_sizer)
    return rebinned


def weight_invert(
    weights: (
        NDArray[np.integer | np.floating | np.complexfloating] | int | float | np.number
    ),
    threshold: float | None = None,
    use_abs: bool = False,
) -> NDArray[np.integer | np.floating | np.complexfloating] | int | float | np.number:
    """
    The weights invert function cleans the weights given by removing
    the values that are 0, NaN or Inf ready for additional calculation.
    If a threshold is set, then we check the values that match the threshold
    instead of checking for zeros.

    Parameters
    ----------
    weights: NDArray[np.integer | np.floating | np.complexfloating] | int | float | np.number
        An array of values of some dtype
    threshold: float | None, optional
        A real number set as the threshold for the array.
        By default its set to None, in this case function checks
        for zeros, by default None
    use_abs: bool, optional
        If True, take the absolute value (sometimes useful for complex numbers)
        By default this is False, so will leave as a complex number and invert, by default False

    Returns
    -------
    result: NDArray[np.integer | np.floating | np.complexfloating] | int | float | np.number
        The weights array that has had NaNs and Infinities removed, and zeros OR
        values that don't meet the threshold.
    """
    # IDL is able to treat one number as an array (because every number is aprrently an array of size 1?)
    # As such we need to check if it's a number less than or equal to 0 and make a zeros array of size 1
    if np.isscalar(weights):
        result = np.zeros(1, dtype=type(weights))
        weights = np.array([weights], dtype=type(weights))
    else:
        result = np.zeros_like(weights, dtype=weights.dtype)
    weights_use = weights
    if use_abs or np.iscomplexobj(weights_use):
        """
        Python and IDL use the where function on complex numbers differently.
        On Python, if you apply a real threshold, it applies to only the real numbers,
        and if you apply an imaginary threshold it applies to only imaginary numbers.
        For example Python:
        test = np.array([1j, 2 + 2j, 3j])
        np.where(test >= 2) == array([1])
        np.where(test >= 2j) == array([1,2])
        Meanwhile in IDL:
        test = COMPLEX([0,2,0],[1,2,3]) ;COMPLEX(REAL, IMAGINARY)
        where(test ge 2) == [1, 2]
        where(test ge COMPLEX(0,2)) == [1, 2]

        IDL on the otherhand, uses the ABS function on COMPLEX numbers before using WHERE.
        Hence the behaviour we're seeing above. This is why we also check for a complexobj
        in the if statement
        """
        weights_use = np.abs(weights)

    # If threshold has been set then...
    if threshold is not None:
        # Get the indexes which meet the threshold
        # As indicated before IDL applies abs before using where to complex numbers
        i_use = np.where(weights_use >= threshold)
    else:
        # Otherwise get where they are not zero
        i_use = np.nonzero(weights_use)

    if np.size(i_use) > 0:
        result[i_use] = 1 / weights[i_use]

    # Replace all NaNs with Zeros
    if np.size(np.where(np.isnan(result))) != 0:
        result[np.where(np.isnan(result))] = 0
    # Replace all Infinities with Zeros
    if np.size(np.where(np.isinf(result))) != 0:
        result[np.where(np.isinf(result))] = 0

    # If the result is an array containing 1 result, then return the result, not an array
    if np.size(result) == 1:
        return result[0]
    return result


def array_match(
    array_1: NDArray[np.integer | np.floating | np.complexfloating],
    value_match: NDArray[np.integer | np.floating | np.complexfloating],
    array_2: NDArray[np.integer | np.floating | np.complexfloating] | None = None,
) -> NDArray[np.int64]:
    """
    Find the indices of the input array which match the array of values provided.
    If a second input array is provided, then find the indices where the values
    provided match in both input arrays. This matching can only be done between
    integer values.

    Parameters
    ----------
    array_1: NDArray[np.integer | np.floating | np.complexfloating]
        The input array to match the values in
    value_match: NDArray[np.integer | np.floating | np.complexfloating]
        The values to find in the input array
    array_2: NDArray[np.integer | np.floating | np.complexfloating] | None, optional
        A second array to match the values in via a logical AND, by default is None

    Returns
    -------
    match_indices: NDArray[np.int64]
        The indices of the input array(s) which match the values provided

    Raises
    ------
    ValueError
        Gets raised if value_match is None
    """

    if value_match is None:
        raise ValueError("Value Match Should be a value not None")
    # If array_2 has been supplied, compare which mins and maxes to use based on two arrays
    if array_2 is not None and np.size(array_2) > 0:
        min_use = np.min([np.min(array_1), np.min(array_2)])
        max_use = np.max([np.max(array_1), np.max(array_2)])
        # Also compute the histogram for array_2
        hist2, _, ri2 = histogram(array_2, min=min_use, max=max_use)
    else:
        # If the second array wasn't supplied
        min_use = np.min(array_1)
        max_use = np.max(array_1)
        # Supply a second hist
        hist2 = np.arange(max_use - min_use + 1)
    # Get the histogram for the first
    hist1, _, ri1 = histogram(array_1, min=min_use, max=max_use)
    # Arrays should be the same size, does addition
    hist_combined = hist1 + hist2
    bins = np.where(hist_combined > 0)

    # Select the values to be used
    hist_v1, bins_v1, _ = histogram(bins + min_use)
    omin = bins_v1[0]
    omax = bins_v1[-1]
    hist_v2, _, _ = histogram(value_match, min=omin, max=omax)
    vals = np.nonzero(np.bitwise_and(hist_v1, hist_v2))[0] + omin - min_use
    n_match = vals.size

    if n_match == 0:
        return -1

    ind_arr = np.zeros_like(array_1)
    for vi in range(n_match):
        i = vals[vi]
        if hist1[i] > 0:
            ind_arr[ri1[ri1[i] : ri1[i + 1]]] = 1
        if hist2[i] > 0:
            ind_arr[ri2[ri2[i] : ri2[i + 1]]] = 1

    match_indices = np.nonzero(ind_arr)[0]
    # Return our matching indices
    return match_indices


def meshgrid(
    dimension: int, elements: int, axis: int | None = None, return_integer: bool = False
) -> NDArray[np.int64 | np.float64]:
    """
    Generates a 2D array of X or Y values. Could be replaced by a another function

    Parameters
    ----------
    dimension: int
        Sets the column size of the array to return
    elements: int
        Sets the row size of the array to return
    axis = int | None, optional
        Set axis = 1 for X values, set axis = 2 for Y values, by default is None
    return integer: bool, optional
        dtype is implied by dimension and/or elements
        by default. If True, sets return array to int, by default is False

    Returns
    -------
    result: NDArray[np.int64 | np.float64]
        A numpy array of shape (elements, dimension) that is
        a modified np.arange(elements * dimension).
    """
    if axis is None:
        axis = elements
        elements = dimension
    # If elements is set as 2
    if axis == 2:
        # Replicate LINDGEN by doing arange and resizing, result is floored
        result = np.reshape(
            np.floor(np.arange(elements * dimension) / dimension), (elements, dimension)
        )
    else:
        # Replicate LINDGEN by doing arange and resizing
        result = np.reshape(
            np.arange(elements * dimension) % dimension, (elements, dimension)
        )
    # If we need to return a float, then set the result and return
    if return_integer:
        return result.astype("int")
    # Otherwise return as is
    else:
        return result


def deriv_coefficients(n: int, divide_factorial: bool = False) -> NDArray[np.float64]:
    """
    Computes an array of coefficients resulting in taking the
    n-th derivative of a function of the form x^a (a must not
    be a positive integer less than n)

    Parameters
    ----------
    n: int
        Decides the length of coefficients
    divide_factorial: bool, optional
        Determine if we need to divide by the factorial, by default is False

    Returns
    -------
    coeff: NDArray[np.float64]
        An array of coefficients
    """
    if n <= 0:
        return 0
    # Set up the array
    coeff = np.zeros(n)
    # Set the first coefficient to 1
    coeff[0] = 1
    # For every coefficient
    for m in range(1, n):
        # Had to do m + 1 for the range as IDL coeff[1:1] == coeff[1], but Python coeff[1:1] == array([])
        coeff[1 : m + 1] += -m * coeff[0:m]
    # If we are to divide by the factorial do that to each coefficient
    if divide_factorial:
        for m in range(n):
            coeff[m] /= factorial(m + 1)

    # Return coefficients
    return coeff


def idl_argunique(
    arr: NDArray[np.integer | np.floating | np.complexfloating],
) -> NDArray[np.int64]:
    """
    In IDL the UNIQ function returns the indexes of the unique values within
    an array (that is assumed to be sorted beforehand). In NumPy they use the first
    unique index when using return index, where as in IDL they use the last index.

    Parameters
    ----------
    arr : NDArray[np.integer | np.floating | np.complexfloating]
        A sorted numpy array of any type.

    Returns
    -------
    NDArray[np.int64]
        An array containing indexes of the last occurence of the unique value in arr

    Examples
    --------
    >>> test = np.array([10, 30, 5, 100, 200, 75, 200, 100, 30])
    >>> idl_argunique(test)
    array([0, 1, 3, 4, 6, 8])
    """
    return np.searchsorted(arr, np.unique(arr), side="right") - 1


def angle_difference(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    degree: bool = False,
    nearest: bool = False,
) -> float:
    """
    Calculates the angle difference between two given celestial coordinates.

    Parameters
    ----------
    ra1 : float
        Right Ascension of a coordinate 1 in radians or degrees
    dec1 : float
        Declination of a coordinate 1 in radians or degrees
    ra2 : float
        Right Ascension of a coordinate 2 in radians or degrees
    dec2 : float
        Declination of a coordinate 2 in radians or degrees
    degree : bool, optional
        If True, then we treate the coordinates given as degrees, by default False
    nearest : bool, optional
        Calculate implied angle instead, by default False

    Returns
    -------
    relative_angle : float
        The angle difference in degrees
    """

    if degree:
        unit = u.deg
    else:
        unit = u.rad
    coord1 = SkyCoord(ra=ra1 * unit, dec=dec1 * unit)
    coord2 = SkyCoord(ra=ra2 * unit, dec=dec2 * unit)
    # Use built in astropy separtion instead of calculating it
    relative_angle = coord1.separation(coord2).value
    if nearest:
        return max(relative_angle, 2 * pi - relative_angle)
    else:
        return relative_angle


def parallactic_angle(latitude: float, hour_angle: float, dec: float) -> float:
    """
    Calculates the parallactic angle given latitude (usually a declination), hour_angle and another declination

    Parameters
    ----------
    latitude : float
        An angle in degrees, usually a declination in this package
    hour_angle : float
        The hour angle in degrees
    dec : float
        A declination in degrees

    Returns
    -------
    parallactic_angle : float
        The angle between the great circle through a celestial object and the zenith, and the hour circle of the object
    """

    y_term = np.sin(np.radians(hour_angle))
    x_term = np.cos(np.radians(dec)) * np.tan(np.radians(latitude)) - np.sin(
        np.radians(dec)
    ) * np.cos(np.radians(hour_angle))
    return np.degrees(np.arctan(y_term / x_term))


def simple_deproject_w_term(
    obs: dict,
    params: dict,
    vis_arr: NDArray[np.complex128],
    direction: float,
    logger: Logger,
) -> NDArray[np.complex128]:
    """
    Applies a w-term deprojection to the visibility array

    Parameters
    ----------
    obs : dict
        The observation data structure
    params : dict
        The data from the UVFITS file
    vis_arr : NDArray[np.complex128]
        The visibility array
    direction : float
        The direction we apply to the phase

    Returns
    -------
    vis_arr : NDArray[np.complex128]
        The visibility array with the deprojection applied
    """

    icomp = 1j
    zcen = np.outer(obs["baseline_info"]["freq"], params["ww"])
    sign = 1 if direction > 0 else -1
    phase = np.exp(direction * icomp * zcen)
    for pol_i in range(obs["n_pol"]):
        vis_arr[pol_i, :, :] *= phase

    sign_str = " +1" if sign > 0 else " -1"
    logger.info(f"Applying simple w-term deprojection:{sign_str}")

    return vis_arr


def resistant_mean(
    array: NDArray[np.integer | np.floating | np.complexfloating],
    deviations: int,
    mad_scale: float = 0.67449999999999999,
    sigma_coeff: NDArray[np.float64] = np.array(
        [
            0.020142000000000000,
            -0.23583999999999999,
            0.90722999999999998,
            -0.15404999999999999,
        ]
    ),
) -> int | float | complex | np.number:
    """
    The resistant_mean function translate the IDLAstro function resistant_mean from IDL to Python using NumPy.
    The values mad_scale and sigma_coeff are also retrieved from the same IDLAstro function when running in Double
    Precision Mode.

    The resistant_mean gets the mean of an array which has had a median absolute deviation threshold applied to the
    absolute deviations of the array to exclude outliers.

    If resistant_mean needs to be optimized, it can be vectorized easily enough

    Parameters
    ----------
    array : NDArray[np.integer | np.floating | np.complexfloating]
        A 1 dimensional array of values, multidimensional arrays should be flattened before use
    deviations : int
        The number of median absolute deviations from the median we want use to exclude outliers
    mad_scale : float, optional
        The scale factor for the median absolute deviation, by default 0.67449999999999999
    sigma_coeff : NDArray[np.float64], optional
        The coefficients applied to the polynomial equation to the standard deviation of the points excluded by the outliers for additional exclusion, by default np.array([0.020142000000000000, -0.23583999999999999 , 0.90722999999999998 , -0.15404999999999999])

    Returns
    -------
    resistant_mean : Number
        The mean of the array with outliers excluded using median absolute deviation

    References
    ----------
    .. IDLAstro, RESISTANT_Mean, https://idlastro.gsfc.nasa.gov/ftp/pro/robust/resistant_mean.pro
    """
    # Calculate median of the real part of the array
    median = np.median(array.real)
    # Get the absolute deviation (residuals)
    abs_dev = np.abs(array - median)
    # Calculate Median Absolute Deviation
    # I could have used scipy's median_abs_deviation to get this, but by doing this manually I can guarantee the same behaviour as IDL
    mad = np.median(abs_dev) / mad_scale
    #  Use MAD and the number of deviations
    mad_threshold = deviations * mad
    # Subset the array by the deviations and residuals
    no_outliers = array[np.where(abs_dev <= mad_threshold)]
    # If the deviations is less than 4.5, change the sigma (standard deviation of the subarray) by using a polyval with set sigma coefficient
    # This compensates Sigma for truncation
    # Calculate the standard deviation of the rela and imag separately
    sigma = np.std(no_outliers.real) + np.std(no_outliers.imag) * 1j
    # Set the deviationsX to 1.0 if it's less than 1
    deviationsX = max(deviations, 1.0)
    if deviationsX <= 4.5:
        sigma = sigma / np.polyval(sigma_coeff, deviationsX)
    sigma_threshold = sigma * deviations
    # Use the sigma threshold to again remove outliers from the array
    # Also take the absolute value of sigma_threshold to get the same behaviour as LE in IDL
    subarray = array[np.where(abs_dev <= np.abs(sigma_threshold))]
    # Get the mean of the subset array which contains no outliers
    return np.mean(subarray)


def run_command(cmd: str, dry_run=False):
    """
    Runs the command string `cmd` using `subprocess.run`. Returns any text output to stdout

    Parameters
    ----------
    cmd : str
         The command to run on the command line
    dry_run : bool
         If True, don't actually run the command. Defaults to False (so defaults to running the command)
    """

    if dry_run:
        stdout = "This was a dry run, not launching IDL code\n"
    else:
        stdout = subprocess.run(cmd.split(), stdout=subprocess.PIPE, text=True).stdout

    return stdout


def vis_weights_update(
    vis_weights: NDArray[np.float64], obs: dict, psf: dict | h5py.File, params: dict
) -> tuple[NDArray[np.float64], dict]:
    """
    Update the visibility weights array to match any updates to the observation
    metadata dictionary, including flagged times, frequencies, tiles, and
    min/max baseline length.

    Parameters
    ----------
    vis_weights : NDArray[np.float64]
        Visibility weights array.
    obs : dict
        Observation metadata dictionary
    psf: dict | h5py.File
        Beam dictionary
    params : dict
        Visibility metadata dictionary

    Returns
    -------
    vis_weights : NDArray[np.float64]
        Updated vis_weights given the modified observation metadata dictionary
    obs : dict
        The observation metadata dictionary, now containing the correct summary
        statistics of the new flagging
    """
    kx_arr = params["uu"] / obs["kpix"]
    ky_arr = params["vv"] / obs["kpix"]
    dist_test = np.sqrt(kx_arr**2 + ky_arr**2) * obs["kpix"]
    dist_test = np.outer(obs["baseline_info"]["freq"], dist_test)
    flag_dist_i = np.where(
        (dist_test < obs["min_baseline"]) | (dist_test > obs["max_baseline"])
    )
    conj_i = np.where(ky_arr > 0)
    if conj_i[0].size > 0:
        kx_arr[conj_i] = -kx_arr[conj_i]
        ky_arr[conj_i] = -ky_arr[conj_i]
    psf_dim = psf["dim"]
    if isinstance(psf, h5py.File):
        psf_dim = psf_dim[0]
    xcen = np.outer(obs["baseline_info"]["freq"], kx_arr)
    xmin = np.floor(xcen) + obs["dimension"] / 2 - (psf_dim / 2 - 1)
    ycen = np.outer(obs["baseline_info"]["freq"], ky_arr)
    ymin = np.floor(ycen) + obs["elements"] / 2 - (psf_dim / 2 - 1)

    range_test_x_i = np.where(
        (xmin <= 0) | ((xmin + psf_dim - 1) >= obs["dimension"] - 1)
    )
    if range_test_x_i[0].size > 0:
        xmin[range_test_x_i] = -1
        ymin[range_test_x_i] = -1
    range_test_y_i = np.where(
        (ymin <= 0) | ((ymin + psf_dim - 1) >= obs["elements"] - 1)
    )
    if range_test_y_i[0].size > 0:
        xmin[range_test_y_i] = -1
        ymin[range_test_y_i] = -1
    del range_test_x_i
    del range_test_y_i

    if flag_dist_i[0].size > 0:
        xmin[flag_dist_i] = -1
        ymin[flag_dist_i] = -1

    # If flag_frequencies is false, freq_use should be all 1's anyway, so this shouldn't affect anything
    freq_cut_i = np.where(obs["baseline_info"]["freq_use"] == 0)
    if freq_cut_i[0].size > 0:
        vis_weights[0 : obs["n_pol"], freq_cut_i[0], :] = 0
    tile_cut_i = np.where(obs["baseline_info"]["tile_use"] == 0)
    if tile_cut_i[0].size > 0:
        bi_cut = array_match(
            obs["baseline_info"]["tile_a"],
            tile_cut_i[0] + 1,
            obs["baseline_info"]["tile_b"],
        )
        if np.size(bi_cut) > 0:
            vis_weights[0 : obs["n_pol"], :, bi_cut] = 0

    time_cut_i = np.where(obs["baseline_info"]["time_use"] == 0)[0]
    bin_offset = np.append(obs["baseline_info"]["bin_offset"], kx_arr.size)
    time_bin = np.zeros(kx_arr.size)
    for ti in range(obs["baseline_info"]["time_use"].size):
        time_bin[bin_offset[ti] : bin_offset[ti + 1]] = ti
    for ti in range(time_cut_i.size):
        ti_cut = np.where(time_bin == time_cut_i[ti])
        if ti_cut[0].size > 0:
            vis_weights[0 : obs["n_pol"], :, ti_cut] = 0

    flag_i = np.where(vis_weights[0] <= 0)
    flag_i_new = np.where(xmin < 0)
    if flag_i_new[0].size > 0:
        vis_weights[0 : obs["n_pol"], flag_i_new[0], flag_i_new[1]] = 0
    if flag_i[0].size > 0:
        xmin[flag_i] = -1
        ymin[flag_i] = -1

    if min(np.max(xmin), np.max(ymin)) < 0:
        obs["n_vis"] = 0
        return vis_weights, obs

    bin_n, _, _ = histogram(xmin + ymin * obs["dimension"], min=0)
    obs["n_vis"] = np.sum(bin_n)

    obs["n_time_flag"] = np.sum(1 - obs["baseline_info"]["time_use"])
    obs["n_tile_flag"] = np.sum(1 - obs["baseline_info"]["tile_use"])
    obs["n_freq_flag"] = np.sum(1 - obs["baseline_info"]["freq_use"])

    return vis_weights, obs


def split_vis_weights(
    obs: dict, vis_weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Separate the indices in the visibility array (data/model/res/weights) into
    interleaved time samples, generally called "even" and "odd" depending on
    whether the time index is even or odd. Interleaved data can be used to generate
    cross power spectra and propagate uncertainties, see eq 2 of Jacobs et al. 2016.
    Ensures that the same flagging is applied to both sets.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    vis_weights : NDArray[np.float64]
        Visibility weights array

    Returns
    -------
    vis_weights : NDArray[np.float64]
        Full visibility weights array with the same flagging applied
        to both interleaved sets.
    bi_use  NDArray[np.int64]
        Baseline index arrays for interleaved time samples, separated
        by "even" and "odd" indices.
    """
    # Not always used in vis_noise_calc requires this check in other cases
    if obs["n_time"] < 2:
        return vis_weights

    # Get the number of baselines, should be the last in this case
    nb = vis_weights.shape[-1]
    bin_end = np.zeros(obs["n_time"], dtype=np.int64)
    bin_end[: obs["n_time"] - 1] = (
        obs["baseline_info"]["bin_offset"][1 : obs["n_time"]] - 1
    )
    bin_end[-1] = int(nb - 1)
    bin_i = np.full(nb, -1, dtype=np.int64)
    for t_i in range(obs["n_time"] // 2):
        bin_i[obs["baseline_info"]["bin_offset"][t_i] : bin_end[t_i] + 1] = t_i

    time_start_i = int(np.min(np.nonzero(obs["baseline_info"]["time_use"])[0]))
    nt3 = int(np.floor((obs["n_time"] - time_start_i) / 2) * 2)
    time_use_0 = obs["baseline_info"]["time_use"][time_start_i : time_start_i + nt3 : 2]
    time_use_1 = obs["baseline_info"]["time_use"][
        time_start_i + 1 : time_start_i + nt3 : 2
    ]
    time_use_01 = time_use_0 * time_use_1
    time_use = np.zeros(obs["baseline_info"]["time_use"].shape, dtype=np.int64)
    time_use[time_start_i : time_start_i + nt3 : 2] = time_use_01
    time_use[time_start_i + 1 : time_start_i + nt3 : 2] = time_use_01
    time_cut_i = np.where(time_use <= 0)[0]
    if time_cut_i.size > 0:
        for cut_i in range(time_cut_i.size):
            bin_i_cut = np.where(bin_i == time_cut_i[cut_i])[0]
            if bin_i_cut.size > 0:
                bin_i[bin_i_cut] = -1

    bi_use = [np.where(bin_i % 2 == 0)[0], np.where(bin_i % 2 == 1)[0]]

    # Here we ensure that both even and odd samples are the same size by
    # ensuring both arrays match the smallest size
    if bi_use[0].size < bi_use[1].size:
        bi_use[1] = bi_use[1][0 : bi_use[0].size]
    elif bi_use[1].size < bi_use[0].size:
        bi_use[0] = bi_use[0][0 : bi_use[1].size]

    for pol_i in range(obs["n_pol"]):
        # In IDL they are doing x > y < w < z
        flag_use = np.minimum(
            np.maximum(vis_weights[pol_i, :, bi_use[0]], 0),
            np.minimum(vis_weights[pol_i, :, bi_use[1]], 1),
        )
        # Reset vis_weights
        vis_weights[pol_i] = 0
        # No keywords used for odd_only or even_only in entirety of FHD
        vis_weights[pol_i, :, bi_use[0]] = flag_use
        vis_weights[pol_i, :, bi_use[1]] = flag_use

    return vis_weights, bi_use


def vis_noise_calc(
    obs: dict,
    vis_arr: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    bi_use: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """
    Calculate the noise from the calibrated visibilities by taking
    the difference between the imaginary parts of the interleaved
    time samples. A factor of sqrt(2) is required because there is
    half as many independent noise samples when calculated from just
    the imaginary part.

    Parameters
    ----------
    obs : dict
        The observation metadata dictionary
    vis_arr : NDArray[np.complex128]
        The vsisibility array
    vis_weights : NDArray[np.float64]
        The visibility weights array
    bi_use : NDArray[np.int64] | None, optional
        Baseline index arrays for interleaved time samples, separated
        by "even" and "odd" indices, by default None

    Returns
    -------
    noise_arr: NDArray[np.float64]
        Calculated variance of the noise as a function of polarisation
        and frequency.
    """
    noise_arr = np.zeros([obs["n_pol"], obs["n_freq"]])

    if obs["n_time"] < 2:
        return noise_arr

    if bi_use is None:
        vis_weights_use, bi_use = split_vis_weights(obs, vis_weights)
    else:
        vis_weights_use = vis_weights

    for pol_i in range(obs["n_pol"]):
        data_diff = (
            vis_arr[pol_i, :, bi_use[0]].imag - vis_arr[pol_i, :, bi_use[1]].imag
        )
        vis_weight_diff = np.maximum(
            vis_weights_use[pol_i, :, bi_use[0]], 0
        ) * np.maximum(vis_weights_use[pol_i, :, bi_use[1]], 0)
        for fi in range(obs["n_freq"]):
            ind_use = np.where(vis_weight_diff[:, fi])[0]
            if ind_use.size > 0:
                noise_arr[pol_i, fi] = np.std(data_diff[ind_use, fi]) / np.sqrt(2)

    return noise_arr


def idl_median(
    x: NDArray[np.integer | np.floating | np.complexfloating],
    width: int = 0,
    even: bool = False,
) -> float:
    """
    The IDL Median function doesn't always work as you'd expect, as generally you need to use the
    EVEN keyword to get the median of an even number of elements, otherwise it returns the
    maximum element of the two numbers in the middle of an even sorted array.
    This function replicates the IDL median function, in case you need that functionality.
    Typically though, we recommend using `numpy.median` or `scipy.ndimage.median_filter`.

    Parameters
    ----------
    x : NDArray[np.integer | np.floating | np.complexfloating]
        Data to perform median on
    width : int
        If set, perform a type of median filtering.
    even : bool, optional
        In this case it will just run numpy.median, by default False


    When `width` is set. unfortunately the edge conditions when using cannot be
    replicated soley with `scipy.ndimage.median_filter` so use `median_filter`    and set the edge cases manually

    Returns
    -------
    float
        The median of the array

    See Also
    --------
    numpy.median: Computes the median of an array
    scipy.ndimage.median_filter: Applies a median filter to an array
    """

    if width:
        # IDL median leaves everything within width//2 pixels of the edge alone
        # So just shove the outputs of median_filter everywhere else. None of
        # the `modes` in median_filter capture this behaviour
        output = deepcopy(x)

        hw = width // 2
        output[hw:-hw] = median_filter(x, size=width)[hw:-hw]

        return output

    else:
        if even:
            return np.median(x)
        else:
            med_index = int(np.ceil(len(x) / 2))

            return np.sort(x)[med_index]


def reshape_and_average_in_time(
    vis_array: NDArray[np.complex128],
    n_freq: int,
    n_time: int,
    n_baselines: int,
    vis_weights: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Given a single polarisation 2D `vis_array` of shape (n_freq, n_time*n_baselines),
    reshape into (n_freq, n_time, n_baselines), and then average in time, weighting
    by `vis_weights` (must be of shape (n_freq, n_time, n_baselines))
    Returns the averaged array in shape (n_freq, n_baselines)

    Parameters
    ----------
    vis_array : NDArray[np.complex128]
       The visibility array
    n_freq : int
        Number of frequencies
    n_time : int
        Number of time steps
    n_baselines : int
        Number of baselines
    vis_weights : NDArray[np.float64]
        The visibility weights array

    Returns
    -------
    reshape_array: NDArray[np.complex128]
        The reshaped visibility array the same shape as the visibility weights array
    """

    new_shape = (n_freq, n_time, n_baselines)

    if vis_weights.shape != new_shape:
        exit(
            f"Attempting to use weights with shape {vis_weights.shape} in `reshape_and_average_in_time`, this is not allowed"
        )

    reshape_array = np.reshape(vis_array, new_shape)
    reshape_array = np.sum(reshape_array * vis_weights, axis=1)

    return reshape_array


def region_grow(
    image: NDArray[np.integer | np.floating | np.complexfloating],
    roiPixels: NDArray[np.integer],
    low: int | float | None = None,
    high: int | float | None = None,
) -> NDArray[np.integer | np.floating | np.complexfloating] | None:
    """
    Replicates IDL's Region Grow, where a region of interest will grow based upon a given threshold
    within a 2D array. It finds all the pixels within the array that are connected neighbors via the
    threshold and blob detection using SciPy's `label` function. In this case, the standard deviation
    form of this function hasn't been implemented as pyfhd will only use this function once with a threshold.

    If you want to use standard deviation region growing adjusting the function can be done by potentially
    implementing skimage's blob detection algorithms for the labelling and keeping the rest the same.

    Parameters
    ----------
    image : NDArray[np.integer | np.floating | np.complexfloating]
        A 2D array of pixels
    roiPixels : NDArray[np.integer]
        The region of interest given as FLAT indexes i.e. array.flat
    low : int | float | None, optional
        The low threshold, any number below this is considered background,
        If left as None, this will be the lowest value of the region of interest, by default None
    high : int | float | None, optional
        The high threshold, any number higher than this is considered background,
        If left as None, this will be the highest value of the region of interest, by default None

    Returns
    -------
    growROIPixels: NDArray[np.integer | np.floating | np.complexfloating] | None
        The grown region of interest that has connected neighbours by using the threshold

    See Also
    --------
    scipy.ndimage.label: Labels an image based off a given kernel

    Notes
    -----
    'scikit-image Blob Detection' : https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html
    """
    # Get the roi and set the low and high thresholds if they haven't been so already.
    roi = image.flat[roiPixels]
    if low is None:
        low = np.min(roi)
    if high is None:
        high = np.max(roi)
    # Replace all NaNs with Zeros
    nans = np.isnan(image)
    image[nans] = 0
    # Get all the values that are within the threshold
    threshArray = np.zeros_like(image)
    threshArray[np.where((image >= low) & (image <= high))] = 1
    threshArray[nans] = 0
    # Do binary blob detection with the label function
    labelArray, _ = label(threshArray)
    # Make the edges background as the Region grow IDL does not do the edges
    # of an array
    if labelArray.ndim == 2:
        labelArray[0, :] = 0
        labelArray[:, 0] = 0
        labelArray[-1, :] = 0
        labelArray[:, -1] = 0
    else:
        labelArray[0] = 0
        labelArray[-1] = 0
    # Get the histogram of the labels to ascertain the neighbours we will be interested in
    if np.size(roiPixels) > 1:
        labels, _, _ = histogram(labelArray.flat[roiPixels], min=0)
        labels = np.nonzero(labels != 0)[0]
        nLabels = labels.size
    else:
        nLabels = 1
        labels = labelArray.flat[roiPixels]
        if not isinstance(labels, np.ndarray):
            labels = np.array([labels], dtype=labelArray.dtype)
    # Ignore the first label if it's 0 as it's the background
    if labels[0] == 0:
        nLabels -= 1
        if nLabels > 0:
            labels = labels[1:]
    # The histogram will have a minimum of 1 so we need to take 1 off the labels
    if nLabels:
        labels -= 1
    # Get a histogram of all the labels
    labelHist, _, revInd = histogram(labelArray, min=1)
    # Get the number of pixels we will be growing to
    nPixels = np.sum(labelHist[labels]) if nLabels else 0
    # If we have any pixels to grow, then grow
    if nPixels > 0:
        # Only one label
        if nLabels == 1:
            growROIPixels = revInd[revInd[labels[0]] : revInd[labels[0] + 1]]
        else:
            # Take in all the labels and save all the flat indexes
            growROIPixels = np.empty(nPixels, dtype=np.int64)
            j = 0
            for i in range(nLabels):
                if revInd[labels[i] + 1] <= revInd.size:
                    growROIPixels[j : j + labelHist[labels[i]]] = revInd[
                        revInd[labels[i]] : revInd[labels[i] + 1]
                    ]
                    j = j + labelHist[labels[i]]
    else:
        # Return None if we didn't have anywhere to grow
        growROIPixels = None
    # Return the flat indexes
    return growROIPixels


def crosspol_split_real_imaginary(
    image: NDArray[np.complex128], pol_names: list[str] | None = None
) -> tuple[NDArray[np.complex128], list[str] | None]:
    """
    Reformat the input full polarisation image, containing PP, PQ, QP, and QQ,
    into PP, real(PQ), imaginary(PQ), and QQ. PQ and QP (i.e. XY and YX)
    are complex and conjugate mirrors of one another. To make a understandable
    image, we can plot the real and imaginary parts of PQ separately without
    loss of information.

    Parameters
    ----------
    image : NDArray[np.complex128]
        Image array ordered in polarisation by PP, PQ, QP, and QQ.
    pol_names : list[str] | None, optional
        The name of the polarisations, by default None

    Returns
    -------
    image : NDArray[np.complex128]
        Image array ordered in polarisation by PP, real(PQ), imaginary(PQ),
        and QQ
    pol_names : list[str]|None]
        New polarisation name array to reflect real(PQ) and imaginary(PQ)
    """
    crosspol_image: NDArray[np.complex128] = image[2]
    image[2] = crosspol_image.real
    image[3] = crosspol_image.imag

    if pol_names is not None:
        crosspol_name: str = pol_names[2]
        pol_names[2] = f"{crosspol_name}_real"
        pol_names[3] = f"{crosspol_name}_imag"

    return image, pol_names

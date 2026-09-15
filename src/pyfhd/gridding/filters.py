import h5py
import numpy as np
from numpy.typing import NDArray
from logging import Logger

from ..pyfhd_tools.pyfhd_utils import weight_invert
from . import gridding_utils


def filter_uv_uniform(
    image_uv: NDArray[np.complex128],
    *,
    vis_count: NDArray[np.float64] | None = None,
    obs: dict | None = None,
    psf: dict | h5py.File | None = None,
    params: dict | None = None,
    pyfhd_config: dict | None = None,
    logger: Logger | None = None,
    weights: NDArray[np.float64] | None = None,
    fi_use: NDArray[np.integer] | None = None,
    bi_use: NDArray[np.integer] | None = None,
    mask_mirror_indices: bool = False,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    Perform uniform weighting in {u,v} space.

    Parameters
    ----------
    image_uv : NDArray[np.complex128]
        A 2D {u,v} gridded plane to be filtered
    vis_count : NDArray[np.float64] | None
        2D array of number of contributing visibilities per pixel on the {u,v} grid
    obs : dict | None, optional
        Observation metadata dictionary, only used if vis_count is not passed.
        By default None.
    psf : dict | h5py.File
        Beam metadata dictionary, only used if vis_count is not passed.
        By default None.
    params : dict | None, optional
        Visibility metadata dictionary, only used if vis_count is not passed.
        By default None.
    pyfhd_config : dict | None, optional
        Run option dictionary, by default None
    logger : Logger | None, optional
        pyfhd's logger, by default None
    weights : NDArray[np.float64] | None, optional
        If vis_count is supplied, this is only used if it's the same shape as
        image_uv in which case it's used as a mask when calculating the normalization
        (uv pixels with weights>0 are included in the normalization calculation).
        If vis_count is not supplied and obs, psf, and params are all supplied,
        this must be the vis_weights, shaped like the visibility array. If none
        of vis_count, obs, psf, and params are supplied, this is used to calculate
        vis_count and should be the shape of image_uv.
    fi_use : NDArray[np.integer] | None, optional
        Frequency index array for gridding (used e.g. when gridding all frequencies
        for continuum images), by default None.
    bi_use : NDArray[np.integer] | None, optional
        Baseline index array for gridding, i.e even vs odd time stamps, by default None
    mask_mirror_indices : bool, optional
        Exclude baselines mirrored along the v-axis, by default False

    Returns
    -------
    image_uv_filtered : NDArray[np.complex128]
        The filtered 2D {u,v} plane
    filter_use : NDArray[np.float64]
        The filter used

    Raises
    ------
    TypeError
        In the case obs or params is None and vis_count is also None
    """

    # If you need the name, grab it from pyfhd_config where needed

    # This does not make use of fine-grained flagging, but relies on coarse
    # flags from the obs structure (e.g. a list of tiles completely flagged, and
    # of frequencies completely flagged)
    if vis_count is None:
        if obs is not None and psf is not None and params is not None:
            vis_count = gridding_utils.visibility_count(
                obs=obs,
                psf=psf,
                params=params,
                vis_weights=weights,
                logger=logger,
                fi_use=fi_use,
                bi_use=bi_use,
                mask_mirror_indices=mask_mirror_indices,
            )
        elif weights is not None and np.size(weights) == np.size(image_uv):
            if np.max(weights) == 0:
                vis_count = weights
            else:
                vis_count = np.abs(weights) / np.min(np.abs(weights[weights > 0]))
        else:
            raise ValueError(
                "Cannot determine vis count. Either vis_count must be provided "
                "or all of obs, psf and params must be provided or weights must "
                "be provided and match `image_uv` in size."
            )

    # Get the parts of the filter we're using
    filter_use = weight_invert(vis_count, threshold=1)
    # Get the weights index as well
    if weights is not None and np.size(weights) == np.size(image_uv):
        wts_i = np.nonzero(weights)
    else:
        wts_i = np.nonzero(filter_use)
    # Apply a mean normalization
    if np.size(wts_i) > 0:
        filter_use /= np.mean(filter_use[wts_i])
    else:
        filter_use /= np.mean(filter_use)
    # Return the filtered
    return image_uv * filter_use, filter_use

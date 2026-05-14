import numpy as np
from numpy.typing import NDArray
from pyfhd.pyfhd_tools.pyfhd_utils import weight_invert
import pyfhd.gridding.gridding_utils as gridding_utils
from logging import Logger


def filter_uv_uniform(
    image_uv: NDArray[np.complex128],
    vis_count: NDArray[np.float64] | None,
    obs: dict | None = None,
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
        Observation metadata dictionary, by default None
    params : dict | None, optional
        Visibility metadata dictionary, by default None
    pyfhd_config : dict | None, optional
        Run option dictionary, by default None
    logger : Logger | None, optional
        pyfhd's logger, by default None
    weights : NDArray[np.float64] | None, optional
        The weights array (aka vis_weights), by default None
    fi_use : NDArray[np.integer] | None, optional
        Frequency index array for gridding, i.e. gridding all frequencies for continuum images, by default None
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

    # This does not make use of fine-grained flagging, but relies on coarse flags from the obs structure
    # (i.e. a list of tiles completely flagged, and of frequencies completely flagged)
    if vis_count is None:
        if obs is not None or params is not None:
            vis_count = gridding_utils.visibility_count(
                obs,
                params,
                weights,
                pyfhd_config,
                logger,
                fi_use,
                bi_use,
                mask_mirror_indices,
            )
        elif weights is not None and np.size(weights) == np.size(image_uv):
            if np.max(weights) == 0:
                vis_count = weights
            else:
                vis_count = np.abs(weights) / np.min(np.abs(weights[weights > 0]))
        else:
            raise TypeError("obs and params must not be None when vis_count is None")

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

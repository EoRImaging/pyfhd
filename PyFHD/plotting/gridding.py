import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from PyFHD.plotting.image import quick_image
from logging import Logger


def plot_gridding(
    obs: dict,
    image_uv: NDArray[np.complex128],
    weights_uv: NDArray[np.complex128],
    variance_uv: NDArray[np.float64],
    pyfhd_config: dict,
    model_uv: NDArray[np.complex64] | None,
    logger: Logger,
) -> None:
    """
    Plot the continuum uv-planes for data, model, and weights.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    image_uv : NDArray[np.complex128]
        UV-plane of the gridded data
    weights_uv : NDArray[np.complex128]
        UV-plane of the gridded weights
    variance_uv : NDArray[np.float64]
        UV-plane of the gridded variances
    model_uv : NDArray[np.complex128], optional
        UV-plane of the gridded model
    pyfhd_config : dict
        Run option dictionary
    logger : Logger
        PyFHD's logger for displaying errors and info to the log files
    """

    # Check if image, weights, and variance contain any non-zero elements
    if (
        (not np.any(image_uv != 0))
        or (not np.any(weights_uv != 0))
        or (not np.any(variance_uv != 0))
    ):
        logger.warning(
            "Gridded image, weights, or variance are all zeros. Plotting skipped."
        )
        return

    # Plotting paths for apparent gridded image (weighted gridded data), variance,
    # and apparent model (weighted gridded model)
    obs_id = pyfhd_config["obs_id"]
    grid_plot_dir = Path(pyfhd_config["output_dir"], "plots", "gridding")
    grid_plot_dir.mkdir(parents=True, exist_ok=True)
    save_path_roots = [
        Path(grid_plot_dir, f"{obs_id}_grid_apparent_image"),
        Path(grid_plot_dir, f"{obs_id}_grid_variance"),
        Path(grid_plot_dir, f"{obs_id}_grid_apparent_model"),
    ]

    # Calculate the apparent image, setting inds with zero weights to zero
    apparent_image = np.abs(np.divide(image_uv, weights_uv, where=weights_uv != 0))

    # Calculate the apparent model, setting inds with zero weights to zero
    if model_uv is not None:
        apparent_model = np.abs(np.divide(model_uv, weights_uv, where=weights_uv != 0))

    # Calculate the x- and y-axis values and labels
    xvals = (np.arange(apparent_image.shape[1]) - apparent_image.shape[1] / 2) * obs[
        "kpix"
    ]
    xtitle = "u (wavelengths)"
    yvals = (np.arange(apparent_image.shape[2]) - apparent_image.shape[2] / 2) * obs[
        "kpix"
    ]
    ytitle = "v (wavelengths)"

    # Get the pol_names
    pol_names = obs["pol_names"]

    for pol_i in range(obs["n_pol"]):
        # Plot the apparent image, variance, and the optional model for each polarization

        # Add a suffix to each path
        save_path_pol = [
            path.with_stem(path.stem + "_" + pol_names[pol_i])
            for path in save_path_roots
        ]

        quick_image(
            apparent_image[pol_i, :, :],
            logger=logger,
            xvals=xvals,
            yvals=yvals,
            xtitle=xtitle,
            ytitle=ytitle,
            title="Apparent Gridded Continuum Data " + pol_names[pol_i],
            cb_title="Amplitude (Jy/beam)",
            savefile=save_path_pol[0],
            missing_value=0,
            log=True,
            png=True,
        )
        quick_image(
            variance_uv[pol_i, :, :],
            logger=logger,
            xvals=xvals,
            yvals=yvals,
            xtitle=xtitle,
            ytitle=ytitle,
            title="Gridded Continuum Variance " + pol_names[pol_i],
            cb_title="(Jy/beam)$^2$",
            savefile=save_path_pol[1],
            missing_value=0,
            log=True,
            png=True,
        )

        if model_uv is not None:
            quick_image(
                apparent_model[pol_i, :, :],
                logger=logger,
                xvals=xvals,
                yvals=yvals,
                xtitle=xtitle,
                ytitle=ytitle,
                title="Apparent Gridded Continuum Model " + pol_names[pol_i],
                cb_title="Amplitude (Jy/beam)",
                savefile=save_path_pol[2],
                missing_value=0,
                log=True,
                png=True,
            )

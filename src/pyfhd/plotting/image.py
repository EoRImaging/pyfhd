import logging
import os
from pathlib import Path

from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def truncate_colormap(cmap, *, minval=0.0, maxval=1.0, nseg=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, nseg)),
    )
    return new_cmap


def color_range(count_missing: int = None) -> tuple:
    """
    Define the color range for the image data.

    Parameters
    ----------
    count_missing : int, optional
        Count of missing values, by default None

    Returns
    -------
    tuple
        A tuple containing the color range and the number of colors.
    """

    # Initialize color range
    color_range = [0, 255]
    if count_missing > 0:
        data_color_range = [1, 255]
    else:
        data_color_range = color_range

    data_n_colors = data_color_range[1] - data_color_range[0] + 1

    return data_color_range, data_n_colors


def log_color_calc(
    data: NDArray[np.integer | np.floating | np.complexfloating],
    *,
    data_range: NDArray[np.integer | np.floating] = None,
    color_profile: str = "log_cut",
    log_cut_val: float = None,
    sigma_clip_level: float | None = None,
    min_abs: float = None,
    count_missing: int = None,
    wh_missing: NDArray[np.integer] = None,
    missing_color: int = None,
    invert_colorbar: bool = False,
) -> tuple:
    """
    Translated version of log_color_calc from IDL to Python.

    Parameters
    ----------
    data : NDArray[np.integer | np.floating | np.complexfloating]
        A 2D array of data to be displayed as an image.
        The data can be of type int, float, or complex.
    data_range : NDArray[np.integer | np.floating], optional
        Min/max color bar range, by default [np.nanmin(image), np.nanmax(image)]
    color_profile : str, optional
        Color bar profiles for logarithmic scaling.
        "log_cut", "sym_log", "abs", by default "log_cut"
    log_cut_val : int | float, optional
        Minimum log value to cut at, by default None
    sigma_clip_level : float, optional
        Number of standard deviations to use as clipping threshold, only used it
        log is True. Default is None meaning that true min and max are used.
    min_abs : int | float, optional
        The minimum absolute value for the color bar, by default None
    count_missing : int, optional
        The number of missing values, by default None
    wh_missing : NDArray[np.integer], optional
        The location of the missing values, by default None
    missing_color : int, optional
        The index of the color bar for missing values, by default None
    invert_colorbar : bool, optional
        Invert the color bar, by default False

    Returns
    -------
    data_log_norm : NDArray[np.int | np.float64]
        The normalized data array.
    cb_ticks : NDArray[np.int | np.float64]
        The color bar ticks.
    cb_ticknames : NDArray[np.int | np.float64]
        The color bar tick names.
    """
    # Define valid color profiles
    color_profile_enum = ["log_cut", "sym_log", "abs"]
    if color_profile not in color_profile_enum:
        raise ValueError(
            f"Color profile must be one of: {', '.join(color_profile_enum)}"
        )

    # Handle data_range
    if data_range is None:
        true_range = [np.nanmin(data), np.nanmax(data)]
    else:
        if len(data_range) != 2:
            raise ValueError("data_range must be a 2-element vector")
        if data_range[1] < data_range[0]:
            raise ValueError("data_range[0] must be less than data_range[1]")

    # Handle sym_log profile constraints
    if color_profile == "sym_log" and data_range[0] > 0:
        logger.warning(
            "sym_log profile cannot be selected with an entirely positive data "
            "range. Switching to log_cut"
        )
        color_profile = "log_cut"

    data_color_range, data_n_colors = color_range(count_missing=count_missing)

    # Handle positive values
    wh_pos = np.nonzero(data > 0)
    count_pos = len(wh_pos[0])
    if count_pos > 0:
        min_pos = np.nanmin(data[wh_pos])
    elif data_range is not None and data_range[0] > 0:
        min_pos = data_range[0]
    elif data_range is not None and data_range[1] > 0:
        min_pos = data_range[1] / 10
    else:
        min_pos = 0.01

    # Handle negative values
    wh_neg = np.nonzero(data < 0)
    count_neg = len(wh_neg[0])
    if count_neg > 0:
        max_neg = np.nanmax(data[wh_neg])
    elif data_range is not None and data_range[1] < 0:
        max_neg = data_range[1]
    elif data_range is not None:
        max_neg = data_range[0] / 10
    else:
        max_neg = true_range[0] / 10

    # Handle zero values
    wh_zero = np.nonzero(data == 0)
    count_zero = len(wh_zero[0])

    percent_clipped = None
    if data_range is None:
        if sigma_clip_level is not None:
            if color_profile in ["abs", "sym_log"]:
                # for sym_log, use the abs to set ranges/outliers the same
                # for positive and negative values
                data_scaled = np.log10(np.abs(data))
            else:
                data_scaled = np.log10(wh_pos)
            mean, _, std = sigma_clipped_stats(
                data_scaled, sigma=sigma_clip_level, maxiters=5
            )

            # Calculate lower and upper bounds of display range in log space,
            # symmetric about the mean.
            lower_bound = mean - sigma_clip_level * std
            upper_bound = mean + sigma_clip_level * std

            data_range = [10**lower_bound, 10**upper_bound]

            # Count how many data points are outside the clipping bounds.
            num_clipped = np.sum(
                (data_scaled < lower_bound) | (data_scaled > upper_bound)
            )
            # Convert number of clipped points to a percentage of the total data.
            percent_clipped = 100 * num_clipped / len(data_scaled)

            # Report display range and clipping percentage.
            logger.info(
                f"Sigma clipping of level {sigma_clip_level} applied: "
                "{data_range[0]:.2e} to {data_range[1]:.2e}"
            )
            logger.info(f"Percentage of data clipped: {percent_clipped:.2f}%")
        else:
            data_range = true_range

    # Handle log_cut color profile
    if color_profile == "log_cut":
        if data_range[1] < 0:
            raise ValueError(
                "log_cut color profile will not work for entirely negative arrays."
            )

        if log_cut_val is None:
            if data_range[0] > 0:
                log_cut_val = np.log10(data_range[0])
            else:
                log_cut_val = np.log10(min_pos)

        log_data_range = [log_cut_val, np.log10(data_range[1])]

        # Handle zero values
        if count_zero > 0:
            min_pos_color = 2
            zero_color = 1
        else:
            min_pos_color = 1

        neg_color = 0

        data_log = np.zeros_like(data)
        data_log[wh_pos] = np.log10(data[wh_pos])
        wh_under = np.nonzero(data < 10**log_cut_val)
        if len(wh_under[0]) > 0:
            data_log[wh_under] = log_data_range[0]

        wh_over = np.nonzero(data_log > log_data_range[1])
        if len(wh_over[0]) > 0:
            data_log[wh_over] = log_data_range[1]

        # Normalize data
        data_log_norm = (
            (data_log - log_data_range[0])
            * (data_n_colors - min_pos_color - 1)
            / (log_data_range[1] - log_data_range[0])
            + data_color_range[0]
            + min_pos_color
        )

        if count_neg > 0:
            data_log_norm[wh_neg] = neg_color
        if count_zero > 0:
            data_log_norm[wh_zero] = zero_color

    elif color_profile == "sym_log":
        if data_range[0] >= 0 or data_range[1] <= 0:
            raise ValueError(
                "sym_log color profile requires both negative and positive values in data_range."
            )

        # Calculate the minimum absolute value
        if min_abs is None:
            if count_pos > 0 and count_neg > 0:
                min_abs = min(min_pos, abs(max_neg))
            elif count_pos > 0:
                min_abs = min_pos
            elif count_neg > 0:
                min_abs = abs(max_neg)
            else:
                min_abs = 1.0

        log_data_range = [np.log10(min_abs), np.log10(data_range[1])]

        # Normalize data
        data_log_norm = np.zeros_like(data, dtype=float)
        wh_pos = np.nonzero(data > 0)
        wh_neg = np.nonzero(data < 0)
        wh_zero = np.nonzero(data == 0)

        midpoint = (data_color_range[1] - data_color_range[0]) // 2

        if len(wh_pos[0]) > 0:
            data_log_norm[wh_pos] = (
                (np.log10(data[wh_pos]) - log_data_range[0])
                * (midpoint)
                / (log_data_range[1] - log_data_range[0])
                + data_color_range[0]
                + midpoint
            )

        if len(wh_neg[0]) > 0:
            # Reverse the mapping for negative values
            data_log_norm[wh_neg] = (
                data_color_range[0]
                + midpoint
                - (
                    (np.log10(abs(data[wh_neg])) - log_data_range[0])
                    * midpoint
                    / (log_data_range[1] - log_data_range[0])
                )
            )

        if len(wh_zero[0]) > 0:
            data_log_norm[wh_zero] = data_color_range[0] + midpoint

        # Handle out-of-bounds values
        wh_under = np.nonzero(data_log_norm < data_color_range[0])
        if len(wh_under[0]) > 0:
            data_log_norm[wh_under] = data_color_range[0]

        wh_over = np.nonzero(data_log_norm > data_color_range[1])
        if len(wh_over[0]) > 0:
            data_log_norm[wh_over] = data_color_range[1]

    # Handle abs color profile
    elif color_profile == "abs":
        data_abs = np.abs(data)
        data_log_norm = (data_abs - data_range[0]) * (data_n_colors - 1) / (
            data_range[1] - data_range[0]
        ) + data_color_range[0]

        # Handle out-of-bounds values
        wh_under = np.nonzero(data_log_norm < data_color_range[0])
        if len(wh_under[0]) > 0:
            data_log_norm[wh_under] = data_color_range[0]

        wh_over = np.nonzero(data_log_norm > data_color_range[1])
        if len(wh_over[0]) > 0:
            data_log_norm[wh_over] = data_color_range[1]

    # Handle missing values
    if count_missing > 0:
        data_log_norm[wh_missing] = missing_color

    # Handle invert_colorbar option
    if invert_colorbar:
        data_log_norm = data_color_range[1] - (data_log_norm - data_color_range[0])

    # Generate colorbar ticks and tick names
    if color_profile == "log_cut":
        cb_ticks = np.linspace(data_color_range[0], data_color_range[1], num=5)
        cb_ticknames = [
            f"{10 ** (tick * (log_data_range[1] - log_data_range[0]) / (data_n_colors - 1) + log_data_range[0]):.2g}"
            for tick in cb_ticks
        ]
    elif color_profile == "sym_log":
        pos_ticks = np.linspace(midpoint, data_color_range[1], num=5)
        neg_ticks = np.linspace(data_color_range[0], midpoint, num=5)
        cb_ticks = np.concatenate([neg_ticks, [midpoint], pos_ticks])
        cb_ticknames = (
            [
                f"-{10 ** (log_data_range[1] - (tick - data_color_range[0]) * (log_data_range[1] - log_data_range[0]) / midpoint):.2g}"
                for tick in neg_ticks
            ]
            + ["0"]
            + [
                f"{10 ** ((tick - midpoint) * (log_data_range[1] - log_data_range[0]) / midpoint + log_data_range[0]):.2g}"
                for tick in pos_ticks
            ]
        )
    elif color_profile == "abs":
        cb_ticks = np.linspace(data_color_range[0], data_color_range[1], num=5)
        cb_ticknames = [
            f"{tick * (data_range[1] - data_range[0]) / (data_n_colors - 1) + data_range[0]:.2g}"
            for tick in cb_ticks
        ]

    return data_log_norm, cb_ticks, cb_ticknames, percent_clipped


def quick_image(
    image: NDArray[np.integer | np.floating | np.complexfloating],
    xvals: NDArray[np.integer | np.floating] = None,
    yvals: NDArray[np.integer | np.floating] = None,
    *,
    data_range: NDArray[np.integer | np.floating] = None,
    data_min_abs: float = None,
    sigma_clip_level: float | None = None,
    percentile_clip_level: float | None = None,
    xrange: NDArray[np.integer | np.floating] = None,
    yrange: NDArray[np.integer | np.floating] = None,
    data_aspect: float = None,
    log: bool = False,
    color_profile: str = "log_cut",
    cmap: str | None = None,
    xtitle: str = None,
    ytitle: str = None,
    title: str = None,
    cb_title: str = None,
    note: str = None,
    charsize: int = None,
    xlog: bool = False,
    ylog: bool = False,
    multi_pos: list = None,
    start_multi_params: dict = None,
    alpha: float = None,
    missing_value: int | float | complex = None,
    savefile: str = None,
    png: bool = False,
    eps: bool = False,
    pdf: bool = False,
) -> None:
    """
    Create an image from a 2D array of data, with optional handling for log scaling,
    missing values and sigma clipping to determine display range.
    The image can be saved to a file or displayed on screen.

    1.  Locate missing values in data, replace with NaN and create mask for them,
        and reserve colour index 0 for them if they exist.
    2.  Determine display range for data. If not provided, apply sigma clipping
        to log of data (if log is True, otherwise just data) to determine a more
        representative range that is not skewed by bright point sources or
        calibration artefacts.
    3.  Map data values onto integer colour range [colour_min, 255]. If log is True,
        we take the log of the data before scaling, and the colour range is scaled
        to the log of the data range. Missing values are set to their reserved
        colour index.
    4.  Render the scaled data as an image using matplotlib, with appropriate
        axis labels, title and colourbar. The colourbar tick labels are
        calibrated to show the original data values corresponding to each colour
        index.
    5.  Save the image to a file if a savefile path is provided or if any of the
        format flags (png, pdf, eps) are True. If no savefile path is provided and
        all format flags are False, display the figure on screen.

    Parameters
    ----------
    image : NDArray[np.integer | np.floating | np.complexfloating]
        A 2D array of data to be displayed as an image.
        The data can be of type int, float, or complex.
    xvals : NDArray[np.integer | np.floating], optional
        An array of x-axis values, by default None
    yvals : NDArray[np.integer | np.floating], optional
        An array of y-axis values, by default None
    data_range : NDArray[np.integer | np.floating], optional
        Min/max color bar range, by default [np.nanmin(image), np.nanmax(image)]
    data_min_abs : float, optional
        The minimum absolute value for the color bar, by default None
    sigma_clip_level : float, optional
        Number of standard deviations to use as clipping threshold, only used it
        log is True. Default is None meaning that true min and max are used.
    percentile_clip_level : float, optional
        Percentile level to use for clipping. For example, a value of 1 means that
        the display range will be set to the 1st and 99th percentiles of the data.
        Only used if log is False. Default is None meaning that true min and max
        are used.
    xrange : NDArray[np.integer | np.floating], optional
        The indices (or xvals, if provided) to zoom the image, by default None
    yrange : NDArray[np.integer | np.floating], optional
        The indices (or yvals, if provided) to zoom the image, by default None
    data_aspect : int | float, optional
        The aspect ratio of y to x, by default None
    log : bool, optional
        Color bar on logarithmic scale, by default False
    color_profile : str, optional
        Color bar profiles for logarithmic scaling.
        "log_cut", "sym_log", "abs", by default "log_cut"
    cmap : str, optional
        Matplotlib colormap to use.
    xtitle : str, optional
        The label for the x-axis.
    ytitle : str, optional
        The label for the y-axis.
    title : str, optional
        The title of the image.
    cb_title : str, optional
        The title of the colourbar.
    note : str, optional
        A small note to place on the bottom right of the image, by default None
    charsize : int, optional
        The size of the font, by default None
    xlog : bool, optional
        Use logarithmic scale for the x-axis, by default False
    ylog : bool, optional
        Use logarithmic scale for the y-axis, by default False
    savefile : str | Path, optional
        The path to save the image file. If None, the image is displayed on screen.
    missing_value : float | int, optional
        The value in the data array that represents missing data. If None, no missing data handling is performed.
    log : bool, optional
        Whether to apply logarithmic scaling to the data. Default is True.
    png : bool, optional
        Whether to save the image as a PNG file. Default is False.
    pdf : bool, optional
        Whether to save the image as a PDF file. Default is False.
    eps : bool, optional
        Whether to save the image as an EPS file. Default is False.

    Returns
    -------
    None
        The function saves the image to a file or displays it on screen.
    """
    # Validate the image input
    if image is None or not isinstance(image, np.ndarray):
        print("Image is undefined or not a valid numpy array.")
        return

    # Ensure the image is 2D
    if image.ndim != 2:
        print("Image must be 2-dimensional.")
        return

    # Handle complex images. Default is to show the real part.
    if np.iscomplexobj(image):
        print("Image is complex, showing real part.")
        image = np.real(image)

    # Handle missing values by setting them to NaN
    if missing_value is not None:
        wh_missing = np.nonzero(image == missing_value)
        count_missing = len(wh_missing[0])
        if count_missing > 0:
            image[wh_missing] = np.nan
        missing_color = 0
    else:
        count_missing = 0
        wh_missing = None
        missing_color = None

    # Validate that 2-value inputs are only 2 values
    if data_range is not None:
        if not isinstance(data_range, np.ndarray | list) or len(data_range) != 2:
            raise ValueError("data_range must be an array with exactly two values.")
    if xrange is not None:
        if not isinstance(xrange, np.ndarray | list) or len(xrange) != 2:
            raise ValueError("xrange must be an array with exactly two values.")
    if yrange is not None:
        if not isinstance(yrange, np.ndarray | list) or len(yrange) != 2:
            raise ValueError("yrange must be an array with exactly two values.")

    # Apply logarithmic scaling if set. This modifies the image input directly
    # to be logarithmically scaled in the color bar range.
    if log:
        image, cb_ticks, cb_ticknames, percent_clipped = log_color_calc(
            data=image,
            data_range=data_range,
            sigma_clip_level=sigma_clip_level,
            color_profile=color_profile,
            log_cut_val=None,
            min_abs=data_min_abs,
            count_missing=count_missing,
            wh_missing=wh_missing,
            missing_color=missing_color,
            invert_colorbar=False,
        )
    else:
        # Apply linear scaling by default. This modifies the image input directly
        # to be linearly scaled in the color bar range.
        if data_range is None:
            if percentile_clip_level is not None:
                lower_bound = np.nanpercentile(image, percentile_clip_level)
                upper_bound = np.nanpercentile(image, 100 - percentile_clip_level)

                data_range = [lower_bound, upper_bound]

                # Count how many data points are outside the clipping bounds.
                num_clipped = np.sum((image < lower_bound) | (image > upper_bound))
                # Convert number of clipped points to a percentage of the total data.
                percent_clipped = 100 * num_clipped / image.size

                logger.info(
                    f"Percentile clipping of level {percentile_clip_level} "
                    "applied: {data_range[0]:.2e} to {data_range[1]:.2e}"
                )
                logger.info(f"Percentage of data clipped: {percent_clipped:.2f}%")
            else:
                data_range = [np.nanmin(image), np.nanmax(image)]

        data_color_range, data_n_colors = color_range(count_missing=count_missing)

        # Find out-of-bounds values
        wh_low = np.nonzero(image < data_range[0])
        wh_high = np.nonzero(image > data_range[1])

        # Scale image data to be in the color range
        image = (image - data_range[0]) * (data_n_colors - 1) / (
            data_range[1] - data_range[0]
        ) + data_color_range[0]

        # Handle out-of-bounds values
        if wh_low[0].size > 0:
            image[wh_low] = data_color_range[0]
        if wh_high[0].size > 0:
            image[wh_high] = data_color_range[1]

        # Handle missing values
        if missing_value is not None and count_missing > 0:
            image[wh_missing] = missing_color

        cb_ticks = np.linspace(data_color_range[0], data_color_range[1], num=5)
        cb_ticknames = [
            f"{tick * (data_range[1] - data_range[0]) / (data_n_colors - 1) + data_range[0]:.2g}"
            for tick in cb_ticks
        ]

    # Set up the plot
    fig, ax = plt.subplots()
    if cmap == "idl":
        cmap = plt.get_cmap("Spectral_r")
        cmap = truncate_colormap(cmap, minval=(20 / 255), maxval=1, nseg=256)
    elif cmap is None:
        if log and color_profile == "sym_log":
            cmap = "RdBu"
        else:
            cmap = "viridis"
    else:
        cmap = plt.get_cmap(cmap)

    # Set up the x and y ranges
    extent = None
    if xvals is not None and yvals is not None:
        # Default extent based on full xvals and yvals
        extent = [xvals[0], xvals[-1], yvals[0], yvals[-1]]
        # Apply xrange to crop the image and adjust extent
        if xrange is not None:
            x_indices = np.logical_and(xvals >= xrange[0], xvals <= xrange[1])
            image = image[:, x_indices]
            xvals = xvals[x_indices]  # Update xvals to match cropped image
            extent[0], extent[1] = xrange[0], xrange[1]
        # Apply yrange to crop the image and adjust extent
        if yrange is not None:
            y_indices = np.logical_and(yvals >= yrange[0], yvals <= yrange[1])
            image = image[y_indices, :]
            yvals = yvals[y_indices]  # Update yvals to match cropped image
            extent[2], extent[3] = yrange[0], yrange[1]
    elif xrange is not None and yrange is not None:
        # If xvals and yvals are not provided, use xrange and yrange directly
        extent = [xrange[0], xrange[1], yrange[0], yrange[1]]
        image = image[np.ix_(yrange, xrange)]

    im = ax.imshow(
        image,
        extent=extent,
        aspect=data_aspect or "auto",
        cmap=cmap,
        vmin=0,
        vmax=255,
        alpha=alpha,
        origin="lower",
    )

    # Add titles and labels
    if title:
        ax.set_title(title, fontsize=charsize or 12)
    if xtitle:
        ax.set_xlabel(xtitle, fontsize=charsize or 10)
    if ytitle:
        ax.set_ylabel(ytitle, fontsize=charsize or 10)

    # Handle logarithmic axes
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    # if log:
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_ticknames)
    if cb_title:
        cbar.set_label(cb_title, fontsize=charsize or 10)

    # Add note about clipping if clipping was applied, including the percentage of data that was clipped.
    if sigma_clip_level is not None or percentile_clip_level is not None:
        clip_note = f"{percent_clipped:.2f}% of data clipped."
        if note is not None:
            note += " " + clip_note
        else:
            note = clip_note

    # Add note if provided
    if note:
        plt.figtext(
            0.99, 0.02, note, horizontalalignment="right", fontsize=charsize or 8
        )

    # Multi-panel plotting
    if multi_pos is not None:
        if len(multi_pos) != 4:
            raise ValueError(
                "multi_pos must be a 4-element list defining the plot position."
            )
        ax.set_position(multi_pos)

    # Handle start_multi_params for multi-panel layout
    if start_multi_params is not None:
        nrows = start_multi_params.get("nrow", 1)
        ncols = start_multi_params.get("ncol", 1)
        index = start_multi_params.get("index", 1) - 1  # Convert to 0-based index
        ax.set_position(
            [
                (index % ncols) / ncols,
                1 - (index // ncols + 1) / nrows,
                1 / ncols,
                1 / nrows,
            ]
        )

    _save_or_display(fig, savefile, png, pdf, eps)


def _save_or_display(fig, savefile, png, pdf, eps):
    """
    Save the figure to a file if a savefile path is provided or if any of the
    format flags (png, pdf, eps) are True. If no savefile path is provided and
    all format flags are False, display the figure on screen.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the rendered image.
    savefile : str | Path | None
        Base path for saving the image file. Defaults to "quick_image" if a
        format flag is set but no path is given.
    png : bool
        Whether to save the image as a PNG file. Default is False.
    pdf : bool
        Whether to save the image as a PDF file. Default is False.
    eps : bool
        Whether to save the image as an EPS file. Default is False.
    """
    # Determine if the output is to be saved to disk
    pub = bool(savefile or png or eps or pdf)

    # Handle file extension and output format
    if pub:
        if not (png or eps or pdf):
            if savefile:
                # Convert savefile to a Path object if it's a string
                savefile = Path(savefile) if isinstance(savefile, str) else savefile
                extension = savefile.suffix.lower()
                if extension == ".eps":
                    eps = True
                elif extension == ".png":
                    png = True
                elif extension == ".pdf":
                    pdf = True
                else:
                    print("Unrecognized extension, using PNG")
                    png = True

        # Set default savefile if not provided
        if not savefile:
            savefile = "quick_image"
            logger.info(
                "No filename specified for quick_image output. Using "
                f"{os.getcwd()}/{savefile}"
            )

        # Ensure only one output format is set
        formats_set = sum([png, eps, pdf])
        if formats_set > 1:
            print("Only one of eps, png, pdf can be set. Defaulting to png.")
            eps = pdf = False
            png = True

        # Append the appropriate file extension
        if isinstance(savefile, Path):
            if png:
                savefile = savefile.with_suffix(".png")
            elif pdf:
                savefile = savefile.with_suffix(".pdf")
            elif eps:
                savefile = savefile.with_suffix(".eps")
        elif isinstance(savefile, str):
            if png:
                savefile += ".png"
            elif pdf:
                savefile += ".pdf"
            elif eps:
                savefile += ".eps"
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

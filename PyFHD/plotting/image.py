import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.stats import sigma_clipped_stats
from logging import Logger


def quick_image(
        data,
        logger: Logger,
        data_range=None,
        xvals=None,
        yvals=None,
        xtitle=None,
        ytitle=None,
        title=None,
        cb_title=None,
        savefile=None,
        missing_value=None,
        log=True,
        sigma_clip_level=3,
        percentile_clip_level=1,
        png=False,
        pdf=False,
        eps=False,
    ):
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
    data : NDArray[np.integer | np.floating | np.complexfloating]
        A 2D array of data to be displayed as an image. The data can be of type int, float, or complex.
    data_range : NDArray[np.integer | np.floating], optional
        The range of values to display. If None, the min and max of the data are used.
    sigma_clip_level : float, optional
        Number of standard deviations to use as clipping threshold. Default is 3.
        Pass None to skip clipping and use true min/max instead.
    percentile_clip_level : float, optional
        Percentile level to use for clipping. For example, a value of 1 means that
        the display range will be set to the 1st and 99th percentiles of the data.
        Default is 1. Pass None to skip clipping and use true min/max instead.
    xvals : NDArray[np.floating], optional
        The x-axis values corresponding to the columns of the data array.
    yvals : NDArray[np.floating], optional
        The y-axis values corresponding to the rows of the data array.
    xtitle : str, optional
        The label for the x-axis.
    ytitle : str, optional
        The label for the y-axis.
    title : str, optional
        The title of the image.
    cb_title : str, optional
        The title of the colourbar.
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
    
    # --- Handle missing values ---
    # Identify missing pixels, create mask for them and replace with NaN so they can be handled separately in plotting.
    # We also determine how many missing pixels there are, and whether we need to reserve colour index 0 for them.
    if missing_value is not None:
        missing_mask = data == missing_value  # Create a boolean mask where True indicates missing values
        num_missing = np.sum(missing_mask)  # Count number of missing values by summing True values in the mask
        if num_missing > 0:
            data = data.copy()  # Create copy of data to avoid modifying original
            data[missing_mask] = np.nan   # Set missing values to NaN so they can be handled separately in plotting
            colour_min = 1  # Reserve colour index 0 for missing values
        else:  # If no missing values, keep original colour range
            colour_min = 0 
    else:  # No missing value specified, so no special handling for missing data.
        missing_mask = None
        num_missing = 0
        colour_min = 0


    # --- Calculate data range ---
    # Determine the display range for the data. If not provided, we can apply sigma or percentile clipping to the data to determine
    # a more representative range that is not skewed by bright point sources or calibration artefacts.
    # Also calculate and report the percentage of data that is clipped by this process.
    
    if data_range is None:
        if log:
            positive_data = data[data > 0]  # Creates array containing only positive values from data. This is necessary because we cannot take the log of non-positive values.
            if len(positive_data) == 0:  # If there are no positive values, we cannot apply logarithmic scaling, so we raise an error.
                logger.warning(
                   "Data must contain positive values for logarithmic scaling."
                )
    
    if sigma_clip_level is not None and log:  # If sigma clipping is requested
        data_scaled = np.log10(positive_data)  # Take log of positive data to get distribution in log space

        # Use sigma clipping to calculate mean and standard deviation of log data, which will be used to determine clipping bounds.
        # maxiters=5 means we will perform up to 5 iterations of clipping to refine the statistics.
        mean, median, std = sigma_clipped_stats(data_scaled, sigma=sigma_clip_level, maxiters=5)  

        # Calculate lower and upper bounds of display range in log space, symmetric about the mean. 
        lower_bound = mean - sigma_clip_level * std
        upper_bound = mean + sigma_clip_level * std

        data_range = [10 ** lower_bound, 10 ** upper_bound]  # Convert bounds back to linear data units (from log space).
        
        num_clipped = np.sum((data_scaled < lower_bound) | (data_scaled > upper_bound))  # Counts how many data points are outside the clipping bounds in log space.
        percent_clipped = 100 * num_clipped / len(data_scaled)  # Converts number of clipped points to a percentage of the total data.
        
        # Report display range and clipping percentage.
        logger.info(
            f"Sigma clipping of level {sigma_clip_level} applied: {data_range[0]:.2e} to {data_range[1]:.2e}"
        )
        logger.info(
            f"Percentage of data clipped: {percent_clipped:.2f}%"
        )

    elif percentile_clip_level is not None and not log:  # If percentile clipping is requested (only applied in linear space)
        lower_bound = np.nanpercentile(data, percentile_clip_level)  # Calculate lower bound as the value below which a certain percentage of the data falls.
        upper_bound = np.nanpercentile(data, 100 - percentile_clip_level)  # Calculate upper bound as the value above which a certain percentage of the data falls.

        data_range = [lower_bound, upper_bound]

        num_clipped = np.sum((data < lower_bound) | (data > upper_bound))  # Count how many data points are outside the clipping bounds.
        percent_clipped = 100 * num_clipped / data.size  # Convert number of clipped points to a percentage of the total data.

        logger.info(
            f"Percentile clipping of level {percentile_clip_level} applied: {data_range[0]:.2e} to {data_range[1]:.2e}"
        )
        logger.info(
            f"Percentage of data clipped: {percent_clipped:.2f}%"
        )

    else:  # If no clipping is requested, we can use the full range of the data. If log scaling is applied, we should consider the range of the positive data since non-positive values cannot be displayed on a log scale.
        if log:
            data_range = [np.min(positive_data), np.max(positive_data)]  # If applying log scaling, we should only consider the range of the positive data since non-positive values cannot be displayed on a log scale.
        else:
            data_range = [np.nanmin(data), np.nanmax(data)] # If not applying log scaling, we can consider the full range of the data including any non-positive values.
        logger.info(
            f"Using full data range: {data_range[0]:.2e} to {data_range[1]:.2e}"
        )


    # --- Apply logarithmic scaling and map to colour range ---
    # Log-transform and scale data to colour range (colour_min - 255), and set missing values to colour 0.
    colour_max = 255
    # Calculate number of usable colours in the colour map. If missing pixels exist, usable colour range starts from 1 instead of 0.
    num_colours = colour_max - colour_min + 1

    # Calculate log of data range bounds.The lower bound corresponds to colour_min and the upper bound corresponds to colour_max.
    if log:
        data_min = np.log10(data_range[0])
        data_max = np.log10(data_range[1])

        # Take log of data, with non-positive values set to data_min so they land at the bottom of the colour scale.
        data_clipped = np.full_like(data, data_min, dtype=float)  # Initalise array for log of data.
        positive_indices = data > 0  # Create boolean mask for where data is positive.
        # Take the log of the positive values and leave non-positive values at data_min.
        data_clipped[positive_indices] = np.log10(data[positive_indices])

    else:  # If not applying log scaling, we can skip straight to scaling the data to the colour range.
        data_clipped = data
        data_min = data_range[0]
        data_max = data_range[1]

    # Clip data to [data_min, data_max] so that any values outside the display range land at the ends of the colour scale rather than wrapping around.
    data_clipped = np.clip(data_clipped, data_min, data_max)

    # Linearly stretch log of data from [data_min, data_max] onto [colour_min, colour_max] to get array of colour indices corresponding to data values.
    data_scaled = (data_clipped - data_min) * num_colours / (data_max - data_min) + colour_min

    # Set missing pixels to their reserved colour index 0
    if num_missing > 0:
        data_scaled[missing_mask] = 0


    # --- Render image ---
    # Render the image using matplotlib, with appropriate axis labels, title and colourbar.
    
    # The extent is defined as (xmin, xmax, ymin, ymax), where xmin and xmax are the first and last values in xvals, and ymin and ymax are the first and last values in yvals.
    extent = None
    if xvals is not None and yvals is not None:  
        extent = (xvals[0], xvals[-1], yvals[0], yvals[-1])  # Set extent of axes so that ticks show physical coordiantes.

    fig, ax = plt.subplots() # Create a new figure and axes for plotting the image.
    # Render the image using imshow, with the specified colour range, colormap, and axis extent.
    # Set origin='lower' so that the [0,0] index of the data is at the bottom-left of the plot.
    im = ax.imshow(
        data_scaled,
        vmin=0,
        vmax=255,
        cmap='viridis',
        extent=extent,
        origin='lower',
        aspect='auto'
    )
    
    # Set axis labels and title if provided.
    if title:
        ax.set_title(title)
    if xtitle:
        ax.set_xlabel(xtitle)
    if ytitle:
        ax.set_ylabel(ytitle)

    # Add note about clipping if clipping was applied, including the percentage of data that was clipped.
    if sigma_clip_level is not None or percentile_clip_level is not None:
        clip_note = f"{percent_clipped:.2f}% of data clipped."
    
    fig.text(0.01, 0.01, clip_note, fontsize=8)

    # Add colourbar with ticks calibrated to show original data values corresponding to each colour index.
    cbar = plt.colorbar(im, ax=ax) 

    # Create tick positions in colour space (between colour_min and colour_max).
    tick_positions = np.linspace(colour_min, colour_max, 5)

    # Convert tick positions back to data values using the inverse of the scaling transformation.
    tick_data_values = (tick_positions - colour_min) / num_colours * (data_max - data_min) + data_min
    if log:
        # Take the inverse of the log transformation to get back to linear data values to show on the colourbar ticks.
        tick_data_values = 10 ** tick_data_values

    cbar.set_ticks(tick_positions.tolist())  # Set tick positions in colour space
    # Set tick labels to show original data values corresponding to each tick position formatted in scientific notation with 2 significant figures.
    cbar.set_ticklabels([f"{val:.2g}" for val in tick_data_values])

    # Add colourbar title
    if cb_title:
        cbar.set_label(cb_title)


    # --- Save the image to file or display on screen ---
    _save_or_display(fig, savefile, png, pdf, eps)


def _save_or_display(
    fig,
    savefile,
    png,
    pdf,
    eps
    ):
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
    # Determine whether we need to save the file based on whether a savefile path is provided or any format flags are set.
    if savefile or png or pdf or eps:
        # If a format flag is set but no savefile path is given, default to "quick_image" as the base filename.
        if savefile is None:
            savefile = "quick_image"

        # If savefile is a Path object, we can use the with_suffix method to change the file extension based on the format flags.
        if isinstance(savefile, Path):
            if png:
                savefile = savefile.with_suffix(".png")
            elif pdf:
                savefile = savefile.with_suffix(".pdf")
            elif eps:
                savefile = savefile.with_suffix(".eps")
        # If savefile is a string, we can simply append the appropriate extension based on the format flags.
        elif isinstance(savefile, str):
            if png:
                savefile += ".png"
            elif pdf:
                savefile += ".pdf"
            elif eps:
                savefile += ".eps"

        # Save the figure to the specified file with high resolution (300 dpi) and tight bounding box to minimize whitespace around the image.
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:  # No savefile path provided and no format flags set, so we display the figure on screen.
        plt.show()        
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from logging import Logger
import os


def quick_image(
        data,
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
        png=False,
        pdf=False,
        eps=False,
    ):
    """
    Create a logarithmically scaled image.

    Parameters
    ----------
    data : NDArray[np.integer | np.floating | np.complexfloating]
        A 2D array of data to be displayed as an image. The data can be of type int, float, or complex.
    data_range : NDArray[np.integer | np.floating], optional
        The range of values to display. If None, the min and max of the data are used.
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

    # Handle missing values
    if missing_value is not None:
        missing_mask = data == missing_value
        num_missing = np.sum(missing_mask)
        if num_missing > 0:
            data = data.copy()
            data[missing_mask] = np.nan
            colour_min = 1
        else:
            colour_min = 0
    else:
        missing_mask = None
        num_missing = 0
        colour_min = 0
    
    colour_max = 255
    num_colours = colour_max - colour_min + 1
    
    # Handle data_range
    if data_range is None:
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            raise ValueError("Data must contain positive values for logarithmic scaling.")
            return
        data_range = [np.min(positive_data), np.max(positive_data)]

    # Apply log scaling to data range
    log_min = np.log10(data_range[0])
    log_max = np.log10(data_range[1])

    # Apply log scaling to data
    data_log = np.full_like(data, log_min, dtype=float) # Initialize with log_min
    positive_indices = data > 0
    data_log[positive_indices] = np.log10(data[positive_indices])

    # Clip data to range
    data_log = np.clip(data_log, log_min, log_max)

    # Scale data to colour range (colour_min - 255)
    data_scaled = (data_log - log_min) * num_colours / (log_max - log_min) + colour_min

    # Set missing values to colour 0
    if num_missing > 0:
        data_scaled[missing_mask] = 0

    # Determine extent for axes (tells matplotlib what actual coordinate values are)
    extent = None
    if xvals is not None and yvals is not None:
        extent = (xvals[0], xvals[-1], yvals[0], yvals[-1])

    # Display image
    fig, ax = plt.subplots() # Create figure and axis
    im = ax.imshow( 
        data_scaled,
        vmin=0,
        vmax=255,
        cmap='viridis',
        extent=extent,
        origin='lower',
        aspect='auto'
    )
    
    # Add titles and labels
    if title:
        ax.set_title(title)
    if xtitle:
        ax.set_xlabel(xtitle)
    if ytitle:
        ax.set_ylabel(ytitle)

    # Create colourbar with original data values
    cbar = plt.colorbar(im, ax=ax) # Adds color scale to plot

    # Create tick positions in colour space (colour_min - 255)
    tick_positions = np.linspace(colour_min, colour_max, 5)

    # Create tick labels in original data space
    tick_log_values = (tick_positions - colour_min) / num_colours * (log_max - log_min) + log_min
    tick_data_values = 10 ** tick_log_values

    cbar.set_ticks(tick_positions.tolist())
    cbar.set_ticklabels([f"{val:.2g}" for val in tick_data_values])  # Set tick labels

    # Add colourbar title
    if cb_title:
        cbar.set_label(cb_title)

    # Save or show
    if savefile or png or pdf or eps:
        # Determine filename
        if savefile is None:
            savefile = "quick_image"

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

        # # Add extension if needed
        # if png and not savefile.endswith('.png'):
        #     savefile += '.png'
        # elif pdf and not savefile.endswith('.pdf'):
        #     savefile += '.pdf'
        # elif eps and not savefile.endswith('.eps'):
        #     savefile += '.eps'
        # elif not any([savefile.endswith(ext) for ext in ['.png', '.pdf', '.eps']]):
        #     savefile += '.png' # Default to PNG

        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()        
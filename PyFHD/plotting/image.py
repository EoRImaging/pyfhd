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
        A 2D array of data to be displayed as an image.
        The data can be of type int, float, or complex.
    data_range : NDArray[np.integer | np.floating], optional
        The range of values to display.
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


def plot_fits_image(
    fits_file: str,
    output_path: str,
    logger: Logger,
    title: str = "FITS Image",
) -> None:
    """
    Plot a FITS image using Astropy and save it to the specified output directory.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    output_path : str
        Path to output image file.
    title : str, optional
        Title of the plot, by default "FITS Image".
    logger : Logger
        PyFHD's logger for displaying errors and info to the log files

    Returns
    -------
    None
        The function saves the plot to the specified output path.
    """
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Get the data from the first extension
        data = hdul[0].data

        # Check that the data is 2D and non-zero
        if data is None or data.ndim != 2:
            logger.warning(
                f"FITS data must be a 2D array, no image made for {fits_file}."
            )
            return
        if not np.any(data):
            logger.warning(
                f"FITS data array contains only zeros, no image made for {fits_file}."
            )
            return

        # Get the data from the first extension
        header = hdul[0].header

        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"

        # Get units from header
        if "BUNIT" not in header:
            unit = "Jy/str"
        else:
            unit = header["BUNIT"]

        # Create a WCS object for the image
        wcs = WCS(header, relax=True)

        # Calculate the extent of the image in degrees
        ny, nx = data.shape
        x_min, x_max = wcs.wcs_pix2world([0, nx - 1], [0, 0], 0)[0]
        y_min, y_max = wcs.wcs_pix2world([0, 0], [0, ny - 1], 0)[1]

        x_extent = abs(x_max - x_min)  # Extent in degrees along the x-axis
        y_extent = abs(y_max - y_min)  # Extent in degrees along the y-axis

        # Set grid spacing to the extent divided by 4
        min_spacing = 2 * u.deg
        spacing_x = max(x_extent / 4, min_spacing.value) * u.deg
        spacing_y = max(y_extent / 4, min_spacing.value) * u.deg

        # Calculate the percentile-based color bar range
        percentile_range = (1, 99)
        vmin, vmax = np.percentile(data[np.isfinite(data)], percentile_range)

        # Create a figure and axis with WCS projection
        fig, ax = plt.subplots(subplot_kw={"projection": wcs})

        # Plot the image data
        im = ax.imshow(
            data, origin="lower", cmap="gray", aspect="auto", vmin=vmin, vmax=vmax
        )

        # Add a WCS-based grid
        ax.grid(color="white", ls="--", alpha=0.5)
        ax.coords.grid(True, color="white", linestyle="--", alpha=0.5)
        ax.coords[0].set_axislabel("Right Ascension (J2000)")
        ax.coords[1].set_axislabel("Declination (J2000)")

        # Customize tick labels for grid lines with dynamic spacing
        ax.coords[0].set_ticks(spacing=spacing_x, color="white", size=8, width=1)
        ax.coords[0].set_ticklabel(size=10, exclude_overlapping=True)
        ax.coords[1].set_ticks(spacing=spacing_y, color="white", size=8, width=1)
        ax.coords[1].set_ticklabel(size=10, exclude_overlapping=True)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation="vertical")
        cbar.set_label("Flux density (" + unit + ")")

        # Set title
        if title:
            ax.set_title(title)
        elif title is None:
            ax.set_title("FITS Image")

        # Save the plot to the output path
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

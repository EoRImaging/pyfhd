import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from logging import Logger


def plot_fits_image(
    fits_file: str, output_path: str, logger: Logger, title: str = "FITS Image"
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

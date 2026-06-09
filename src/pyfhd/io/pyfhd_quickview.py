import numpy as np
from numpy.typing import NDArray
from pyfhd.io.pyfhd_io import save
from logging import Logger
from pathlib import Path
from astropy.io import fits
from datetime import datetime
from pyfhd.data_setup.obs import update_obs
from pyfhd.beam_setup.beam_utils import beam_image
from pyfhd.pyfhd_tools.unit_conv import pixel_to_radec
from pyfhd.pyfhd_tools.pyfhd_utils import (
    meshgrid,
    rebin,
    weight_invert,
    region_grow,
    crosspol_split_real_imaginary,
)
from pyfhd.gridding.gridding_utils import dirty_image_generate
from pyfhd.plotting.fits_image import plot_fits_image


def get_image_renormalization(
    obs: dict,
    weights: NDArray[np.float64],
    beam_base: NDArray[np.complex128],
    filter_arr: NDArray[np.float64],
    pyfhd_config: dict,
    logger: Logger,
) -> np.ndarray:
    """
    Use the weights to renormalize the image for Jy/beam to Jy/sr. While
    Jy/beam is more common in radio astronomy, Jy/str is a physical
    brightness unit, rather than an instrumental brightness unit.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary.
    weights : NDArray[np.float64]
        Visibility weights array.
    beam_base : NDArray[np.complex128]
        Beam orthoslant image per polarization.
    filter_arr : NDArray[np.float64]
        u-v array of filter weights, relevant for
        a uniform filter.
    pyfhd_config : dict
        pyfhd configuration settings.
    logger : Logger
        pyfhd's Logger.

    Returns
    -------
    NDArray[np.float64]
        Conversion in image space from Jy/beam to Jy/sr per pixel.
    """
    # Use the weights to renormalize the image to units of Jy/sr
    renorm_factor = np.empty(obs["n_pol"])
    for pol_i in range(obs["n_pol"]):
        dirty_image, _, _ = dirty_image_generate(
            weights[pol_i],
            pyfhd_config,
            logger,
            weights=weights[pol_i],
            pad_uv_image=pyfhd_config["pad_uv_image"],
            filter=filter_arr[pol_i],
            beam_ptr=beam_base[pol_i],
            degpix=obs["degpix"],
        )
        dirty_num = dirty_image[obs["dimension"] // 2, obs["elements"] // 2]
        renorm_factor[pol_i] = 1 / dirty_num
        renorm_factor[pol_i] *= (
            beam_base[pol_i, int(obs["obsx"]), int(obs["obsy"])] ** 2
        )
        renorm_factor[pol_i] /= (obs["degpix"] * (np.pi / 180)) ** 2
    return renorm_factor


def quickview(
    obs: dict,
    psf: dict,
    params: dict,
    cal: dict,
    vis_arr: NDArray[np.complex128],
    vis_weights: NDArray[np.float64],
    image_uv: NDArray[np.complex128],
    weights_uv: NDArray[np.complex128],
    variance_uv: NDArray[np.float64],
    uniform_filter_uv: NDArray[np.float64],
    model_uv: NDArray[np.complex128],
    pyfhd_config: dict,
    logger: Logger,
) -> None:
    """
    Generate continuum images from all gridded u-v planes, and save as
    FITS files and optionally PNG files.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary.
    psf : dict
        Beam dictionary.
    params : dict
        Visibility metadata dictionary.
    cal : dict
        Calibration dictionary.
    vis_arr : NDArray[np.complex128]
        Calibrated visibilities array.
    vis_weights : NDArray[np.float64]
        Visibility weights array.
    image_uv : NDArray[np.complex128]
        Continuum uv-plane of the calibrated data.
    weights_uv : NDArray[np.complex128]
        Continuum uv-plane of the weights (the sampling map).
    variance_uv : NDArray[np.float64]
        Continuum uv-plane of the variance (the variance map).
    uniform_filter_uv : NDArray[np.float64]
        Continuum uv-plane of the uniform filter (if used).
    model_uv : NDArray[np.complex128]
        Continuum uv-plane of the model data.
    pyfhd_config : dict
        pyfhd configuration settings.
    logger : Logger
        pyfhd's Logger.
    """
    # Save all the things into the output directory
    pyfhd_config["metadata_dir"] = Path(pyfhd_config["output_dir"], "metadata")
    pyfhd_config["visibilities_path"] = Path(pyfhd_config["output_dir"], "visibilities")
    pyfhd_config["metadata_dir"].mkdir(exist_ok=True)
    pyfhd_config["visibilities_path"].mkdir(exist_ok=True)
    if pyfhd_config["save_obs"]:
        obs_path = Path(
            pyfhd_config["metadata_dir"], f"{pyfhd_config['obs_id']}_obs.h5"
        )
        logger.info(f"Saving the obs dictionary to {obs_path}")
        save(obs_path, obs, "obs", logger=logger)
    if pyfhd_config["save_params"]:
        params_path = Path(
            pyfhd_config["metadata_dir"], f"{pyfhd_config['obs_id']}_params.h5"
        )
        logger.info(f"Saving params dictionary to {params_path}")
        save(params_path, params, "params", logger=logger)
    if pyfhd_config["save_visibilities"]:
        if pyfhd_config["recalculate_grid"]:
            gridding_path = Path(pyfhd_config["output_dir"], "gridding")
            gridding_path.mkdir(exist_ok=True)
            logger.info(f"Saving the gridded uv planes to {gridding_path}")
            save(
                Path(gridding_path, f"{pyfhd_config['obs_id']}_image_uv.h5"),
                image_uv,
                "image_uv",
                logger=logger,
            )
            save(
                Path(gridding_path, f"{pyfhd_config['obs_id']}_weights_uv.h5"),
                weights_uv,
                "weights_uv",
                logger=logger,
            )
            save(
                Path(gridding_path, f"{pyfhd_config['obs_id']}_variance_uv.h5"),
                variance_uv,
                "variance_uv",
                logger=logger,
            )
            save(
                Path(gridding_path, f"{pyfhd_config['obs_id']}_uniform_filter_uv.h5"),
                uniform_filter_uv,
                "uniform_filter_uv",
                logger=logger,
            )
            save(
                Path(gridding_path, f"{pyfhd_config['obs_id']}_model_uv.h5"),
                model_uv,
                "model_uv",
                logger=logger,
            )
        cal_vis_arr_path = Path(
            pyfhd_config["visibilities_path"],
            f"{pyfhd_config['obs_id']}_calibrated_vis_arr.h5",
        )
        logger.info(f"Saving the calibrated visibilities to {cal_vis_arr_path}")
        save(cal_vis_arr_path, vis_arr, "visibilities", logger=logger)
    if pyfhd_config["save_cal"] and pyfhd_config["calibrate_visibilities"]:
        cal_path = Path(pyfhd_config["output_dir"], "calibration")
        cal_path.mkdir(exist_ok=True)
        cal_path = Path(cal_path, f"{pyfhd_config['obs_id']}_cal.h5")
        logger.info(f"Saving the calibration dictionary to {cal_path}")
        save(cal_path, cal, "cal", logger=logger)
    if pyfhd_config["save_weights"]:
        weights_path = Path(
            pyfhd_config["visibilities_path"],
            f"{pyfhd_config['obs_id']}_calibrated_vis_weights.h5",
        )
        logger.info(f"Saving the calibrated weights to {weights_path}")
        save(weights_path, vis_weights, "weights", logger=logger)

    obs_out = update_obs(
        obs, int(obs["dimension"] * pyfhd_config["pad_uv_image"]), obs["kpix"]
    )
    # In case pad_uv_image was a float can get a float out rather than int
    obs_out["dimension"] = int(obs_out["dimension"])
    obs_out["elements"] = int(obs_out["elements"])
    horizon_mask = np.ones([obs_out["dimension"], obs_out["elements"]])
    ra, _ = pixel_to_radec(
        meshgrid(obs_out["dimension"], obs_out["elements"], 1),
        meshgrid(obs_out["dimension"], obs_out["elements"], 2),
        obs_out["astr"],
    )
    horizon_test = np.isnan(ra)
    horizon_mask[horizon_test] = 0

    # Calculate the beam mask and beam indexes associated with that mask
    beam_mask = np.ones([obs_out["dimension"], obs_out["elements"]])
    beam_avg = np.zeros([obs_out["dimension"], obs_out["elements"]])
    beam_base_out = np.empty(
        [obs_out["n_pol"], obs_out["dimension"], obs_out["elements"]]
    )
    beam_correction_out = np.empty_like(beam_base_out)
    for pol_i in range(obs["n_pol"]):
        beam_image_pol = beam_image(psf, obs, pol_i, square=False)
        # Interpolate the beam_image_pol to the new dimensions with padding
        beam_base_out[pol_i] = (
            rebin(beam_image_pol, [obs_out["dimension"], obs_out["elements"]])
            * horizon_mask
        )
        beam_correction_out[pol_i] = weight_invert(beam_base_out[pol_i], 1e-3)
        if pol_i == 0:
            beam_mask_test = beam_base_out[pol_i]
            if pyfhd_config["allow_sidelobe_model_sources"]:
                beam_i = np.where(
                    beam_mask_test.flat >= 0.025
                )  # This is beam_threshold/2 in FHD
            else:
                beam_i = region_grow(
                    beam_mask_test,
                    np.array(
                        [
                            obs_out["dimension"] / 2
                            + obs_out["dimension"] * obs_out["elements"] / 2
                        ],
                        dtype=np.int64,
                    ),
                    low=0.05 / 2,  # This is beam_threshold/2 in FHD
                    high=np.max(beam_mask_test),
                )
            beam_mask0 = np.zeros([obs_out["dimension"], obs_out["elements"]])
            beam_mask0.flat[beam_i] = 1
            beam_avg += beam_base_out[pol_i] ** 2
            beam_mask *= beam_mask0
    beam_avg /= min(obs["n_pol"], 2)
    beam_avg = np.sqrt(np.maximum(beam_avg, 0)) * beam_mask
    beam_i = np.nonzero(beam_mask)

    # Generate our dirty images of the uv planes
    instr_dirty_arr = np.empty([obs["n_pol"], obs["dimension"], obs["elements"]])
    instr_model_arr = np.empty([obs["n_pol"], obs["dimension"], obs["elements"]])
    filter_arr = np.zeros([obs["n_pol"], obs["dimension"], obs["elements"]])
    for pol_i in range(obs["n_pol"]):
        complex_flag = pol_i > 1
        filter = np.empty(0)
        # Get the dirty image of the uv plane and the filter from filter_uv_uniform
        instr_dirty_arr[pol_i], filter, _ = dirty_image_generate(
            image_uv[pol_i],
            pyfhd_config,
            logger,
            uniform_filter_uv=uniform_filter_uv,
            degpix=obs_out["degpix"],
            weights=weights_uv[pol_i],
            pad_uv_image=pyfhd_config["pad_uv_image"],
            filter=filter,
            not_real=complex_flag,
            beam_ptr=beam_base_out[pol_i],
        )
        filter_arr[pol_i] = filter
        instr_model_arr[pol_i], filter, _ = dirty_image_generate(
            model_uv[pol_i],
            pyfhd_config,
            logger,
            uniform_filter_uv=uniform_filter_uv,
            degpix=obs_out["degpix"],
            weights=weights_uv[pol_i],
            pad_uv_image=pyfhd_config["pad_uv_image"],
            filter=filter,
            not_real=complex_flag,
            beam_ptr=beam_base_out[pol_i],
        )
    renorm_factor = get_image_renormalization(
        obs_out, weights_uv, beam_base_out, filter_arr, pyfhd_config, logger
    )
    # Reshape renorm factor to multiply per polarization without loop to [obs["n_pol"], 1, 1]
    renorm_factor = np.expand_dims(renorm_factor.reshape([obs_out["n_pol"], 1]), -1)
    instr_dirty_arr *= renorm_factor
    instr_model_arr *= renorm_factor
    instr_residual_arr = instr_dirty_arr - instr_model_arr
    # Get the pol_names
    pol_names = obs["pol_names"]
    # The cross-polarization XY and YX images are both complex, but are conjugate mirrors of each other
    # To make images of these, we simply take the real and imaginary parts separately
    if obs_out["n_pol"] >= 4:
        logger.info("Peforming Cross Polarization splits for real and imaginary")
        instr_dirty_arr, pol_names = crosspol_split_real_imaginary(
            instr_dirty_arr, pol_names=pol_names
        )
        instr_model_arr, _ = crosspol_split_real_imaginary(instr_model_arr)
        instr_residual_arr, _ = crosspol_split_real_imaginary(instr_residual_arr)
        # The weights should have been saved at this point and we only need them like this from here
        weights_uv, _ = crosspol_split_real_imaginary(weights_uv)

    # Build a fits header
    logger.info("Building the FITS Header for all the FITS files")
    fits_file = fits.PrimaryHDU(instr_dirty_arr[0])
    # Write in the WCS into the header from the astr dictionary in obs_out
    # You may want to change this later to make it more compatible with Astropy directly
    # when eppsilon is switched to python.
    fits_file.header.set("ctype1", str(obs_out["astr"]["ctype"][0]), "Coordinate Type")
    fits_file.header.set("ctype2", str(obs_out["astr"]["ctype"][1]), "Coordinate Type")
    fits_file.header.set(
        "equinox", obs_out["astr"]["equinox"], "Equinox of Ref. Coord."
    )
    cd = obs_out["astr"]["cd"]
    cd[0, :] = cd[0, :] * obs_out["astr"]["cdelt"][0]
    cd[1, :] = cd[1, :] * obs_out["astr"]["cdelt"][1]
    fits_file.header.set("cd1_1", cd[0, 0], "Degrees / Pixel")
    fits_file.header.set("cd1_2", cd[0, 1], "Degrees / Pixel")
    fits_file.header.set("cd2_1", cd[1, 0], "Degrees / Pixel")
    fits_file.header.set("cd2_2", cd[1, 1], "Degrees / Pixel")
    fits_file.header.set(
        "crpix1", int(obs_out["astr"]["crpix"][0]), "Reference Pixel in X"
    )
    fits_file.header.set(
        "crpix2", int(obs_out["astr"]["crpix"][1]), "Reference Pixel in Y"
    )
    fits_file.header.set(
        "crval1", obs_out["astr"]["crval"][0], "R.A. (degrees) of reference pixel"
    )
    fits_file.header.set(
        "crval2", obs_out["astr"]["crval"][1], "Declination of reference pixel"
    )
    fits_file.header.set("pv2_1", obs_out["astr"]["pv2"][0], "Projection parameter 1")
    fits_file.header.set("pv2_2", obs_out["astr"]["pv2"][1], "Projection parameter 2")
    for i in range(obs_out["astr"]["pv1"].size):
        fits_file.header.set(
            f"pv1_{i}", obs_out["astr"]["pv1"][i], "Projection parameters"
        )
    fits_file.header.set(
        "mjd-obs", obs_out["astr"]["mjdobs"], "Modified Julian day of observations"
    )
    fits_file.header.set("date-obs", obs_out["astr"]["dateobs"], "Date of observations")
    fits_file.header.set("radesysa", obs_out["astr"]["radecsys"], "Reference Frame")
    fits_file.header.set(
        "history",
        f"World Coordinate System parameters written by pyfhd: {datetime.now().strftime('%b %d %Y %H:%M:%S')}",
    )
    # Create the fits header to store the dirty images
    fits_file_apparent = fits_file.copy()
    fits_file_apparent.header.set("bunit", "Jy/sr (apparent)")

    # Create the fits header for the weights
    fits_file_uv = fits.PrimaryHDU(np.abs(weights_uv[0]))
    fits_file_uv.header.set("CD1_1", obs["kpix"], "Wavelengths / Pixel")
    fits_file_uv.header.set("CD2_1", 0.0, "Wavelengths / Pixel")
    fits_file_uv.header.set("CD1_2", 0.0, "Wavelengths / Pixel")
    fits_file_uv.header.set("CD2_2", obs["kpix"], "Wavelengths / Pixel")
    fits_file_uv.header.set(
        "CRPIX1", obs_out["dimension"] / 2 + 1, "Reference Pixel in X"
    )
    fits_file_uv.header.set(
        "CRPIX2", obs_out["elements"] / 2 + 1, "Reference Pixel in Y"
    )
    fits_file_uv.header.set("CRVAL1", 0.0, "Wavelengths (u)")
    fits_file_uv.header.set("CRVAL2", 0.0, "Wavelengths (v)")
    fits_file_uv.header.set(
        "MJD-OBS", obs_out["astr"]["mjdobs"], "Modified Julian day of observation"
    )
    fits_file_uv.header.set(
        "DATE-OBS", obs_out["astr"]["dateobs"], "Date of observation"
    )

    # If you need the beam_contour arrays add them here, lines 369-378 in fhd_quickview.pro

    filter_name = pyfhd_config["image_filter"].split("_")[-1]
    fits_output: Path = pyfhd_config["output_dir"] / "fits"
    fits_output.mkdir(exist_ok=True)
    if pyfhd_config["image_plots"]:
        png_output: Path = pyfhd_config["output_dir"] / "plots" / "images"
        png_output.mkdir(exist_ok=True)
    for pol_i in range(obs["n_pol"]):
        logger.info(f"Saving the FITS files for polarization {pol_names[pol_i]}")
        instr_residual = instr_residual_arr[pol_i] * beam_correction_out[pol_i]
        instr_dirty = instr_dirty_arr[pol_i] * beam_correction_out[pol_i]
        instr_model = instr_model_arr[pol_i] * beam_correction_out[pol_i]
        beam_use = beam_base_out[pol_i]

        # Write the fits files for the dirty images
        fits_file_apparent.data = instr_dirty
        instr_dirty_name = (
            f"{pyfhd_config['obs_id']}_{filter_name}_dirty_{pol_names[pol_i]}"
        )
        fits_file_apparent.writeto(
            Path(fits_output, f"{instr_dirty_name}.fits"), overwrite=True
        )
        fits_file_apparent.data = instr_model
        instr_model_name = (
            f"{pyfhd_config['obs_id']}_{filter_name}_model_{pol_names[pol_i]}"
        )
        fits_file_apparent.writeto(
            Path(fits_output, f"{instr_model_name}.fits"), overwrite=True
        )
        fits_file_apparent.data = instr_residual
        instr_residual_name = (
            f"{pyfhd_config['obs_id']}_{filter_name}_residual_{pol_names[pol_i]}"
        )
        fits_file_apparent.writeto(
            Path(fits_output, f"{instr_residual_name}.fits"), overwrite=True
        )
        fits_file.data = beam_use
        beam_name = f"{pyfhd_config['obs_id']}_beam_{pol_names[pol_i]}"
        fits_file.writeto(Path(fits_output, f"{beam_name}.fits"), overwrite=True)
        fits_file_uv.data = np.abs(weights_uv) * obs["n_vis"]
        weights_name = f"{pyfhd_config['obs_id']}_uv_weights_{pol_names[pol_i]}"
        fits_file_uv.writeto(Path(fits_output, f"{weights_name}.fits"), overwrite=True)

        if pyfhd_config["image_plots"]:
            logger.info(
                f"Plotting the continuum images for polarization {pol_names[pol_i]} into {pyfhd_config['output_dir'] / 'plots' / 'images'}"
            )
            plot_fits_image(
                Path(fits_output, f"{instr_dirty_name}.fits"),
                Path(png_output, f"{instr_dirty_name}.png"),
                title=f"Dirty Image {pol_names[pol_i]}",
                logger=logger,
            )
            plot_fits_image(
                Path(fits_output, f"{instr_model_name}.fits"),
                Path(png_output, f"{instr_model_name}.png"),
                title=f"Model Image {pol_names[pol_i]}",
                logger=logger,
            )
            plot_fits_image(
                Path(fits_output, f"{instr_residual_name}.fits"),
                Path(png_output, f"{instr_residual_name}.png"),
                title=f"Residual Image {pol_names[pol_i]}",
                logger=logger,
            )
            plot_fits_image(
                Path(fits_output, f"{beam_name}.fits"),
                Path(png_output, f"{beam_name}.png"),
                title=f"Beam Image {pol_names[pol_i]}",
                logger=logger,
            )
            plot_fits_image(
                Path(fits_output, f"{weights_name}.fits"),
                Path(png_output, f"{weights_name}.png"),
                title=f"Weight Image {pol_names[pol_i]}",
                logger=logger,
            )

import sys
from logging import Logger
from pathlib import Path
import h5py
import importlib_resources
import numpy as np
from numpy.typing import NDArray
from healpy import query_disc
from healpy.pixelfunc import ang2vec, pix2vec, vec2ang
from PyFHD.beam_setup.beam_utils import beam_image
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.gridding.gridding_utils import dirty_image_generate
from PyFHD.io.pyfhd_io import load, save
from PyFHD.pyfhd_tools.pyfhd_utils import (
    angle_difference,
    histogram,
    meshgrid,
    region_grow,
)
from PyFHD.pyfhd_tools.unit_conv import radec_to_altaz, radec_to_pixel
from astropy.coordinates import EarthLocation


def healpix_cnv_apply(
    image: NDArray[np.integer | np.floating | np.complexfloating], hpx_cnv: dict
) -> NDArray[np.float64]:
    """
    healpix_cnv_apply creates a map based off the array/image and healpix convention dictionary given.
    In FHD the healpix_cnv_apply was mainly used as a wrapper for sprsax2, as such I will put the code
    for sprsax2 in here as PyFHD will only use sprsax2 here as we don't have the holographic mapping
    function at this time.

    Vectors 'sa' (interpolation) and 'ija' (index) follow the row-index sparse storage mode as described
    in section 2.7 of Numerical Recipes in C, 2nd edition.

    Parameters
    ----------
    image : NDArray[np.integer | np.floating | np.complexfloating]
        An orthoslant image array that is to be converted to a HEALPix map.
    hpx_cnv : dict
        The HEALPix index/interpolation dictionary

    Returns
    -------
    hpx_map: NDArray[np.float64]
        A flattened 1D HEALPix array with the interpolated values from the image.
    """
    # Is B in sprsax2
    hpx_map = np.zeros(np.size(hpx_cnv["inds"]))
    # Is X in sprsax2 or X_use
    image_vector = image.flatten()
    for i in range(0, np.size(hpx_cnv["i_use"])):
        i_use = hpx_cnv["i_use"][i]
        # Transpose is always True when used inside healpix_cnv_apply when using
        # sprsax2, so we can ignore the transpose keyword
        hpx_map[hpx_cnv["ija"][i]] += hpx_cnv["sa"][i] * image_vector[i_use]
    return hpx_map


def healpix_cnv_generate(
    obs: dict,
    mask: NDArray[np.int64] | None,
    hpx_radius: float,
    pyfhd_config: dict,
    logger: Logger,
    nside: int = None,
) -> dict:
    """
    Generate the HEALPix index/interpolation dictionary that is used to convert
    between orthoslant image pixelization and the RING index HEALPix pixelization
    scheme. Binlinear interpolation is performed to HEALPix pixels centers. All
    angles are in degrees.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    mask : np.ndarray
        If HEALPix indices are not provided in the config file, this boolean
        mask map of the orthoslant image can be used to mask out areas from the
        HEALPix conversion.
    hpx_radius : float
        If HEALPix indices are not provided in the config file, this selects
        a radius in degrees, centred on the RA/Dec of the observation, to be
        converted to HEALPix pixels.
    pyfhd_config : dict
        PyFHD configuration settings
    logger : Logger
        PyFHD's Logger
    nside : int
        The HEALPix nside parameter, calculated based on the observation
        metadata if none provided. Defaults to None.

    Returns
    -------
    dict
        A dictionary containing the orthoslant to HEALPix pixel mapping and
        interpolation. Vectors 'sa' (bilinear interpolation calculated from
        fractional offsets) and 'ija' (flattened index) follow the row-index
        sparse storage mode as described in section 2.7 of Numerical Recipes
        in C, 2nd edition.
    """

    # Have ignored the code that relates to hpx_radius being empty, in PyFHD's case
    # we will assume that a value for hpx_radius is supplied, if you want to add it
    # yourself you can do that here
    hpx_inds = None
    # Fill hpx_inds and nside with values from a file if restrict_healpix_inds has been activated
    if pyfhd_config["restrict_healpix_inds"]:
        if pyfhd_config["healpix_inds"] is None:
            # Get the healpix indexes based off the observation, comes from observation_healpix_inds_select
            files = np.array(
                [
                    {
                        "name": "EoR0_high_healpix_inds.h5",
                        "ra": 0,
                        "dec": -30,
                        "freq": 182,
                    },
                    {
                        "name": "EoR0_low_healpix_inds.h5",
                        "ra": 0,
                        "dec": -30,
                        "freq": 151,
                    },
                    {
                        "name": "EoR1_high_healpix_inds.h5",
                        "ra": 60,
                        "dec": -30,
                        "freq": 182,
                    },
                    {
                        "name": "EoR1_low_healpix_inds.h5",
                        "ra": 60,
                        "dec": -30,
                        "freq": 151,
                    },
                ]
            )
            ang_dist = []
            for file in files:
                ang_dist.append(
                    np.abs(
                        angle_difference(
                            obs["obsra"],
                            obs["obsdec"],
                            file["ra"],
                            file["dec"],
                            degree=True,
                            nearest=True,
                        )
                    )
                )
            ang_dist = np.array(ang_dist)
            ang_use = np.min(ang_dist)
            i_use = np.where(np.abs(ang_dist - ang_use) <= 1)
            files = files[i_use]
            freq_dist = []
            for file in files:
                freq_dist.append(file["freq"])
            freq_dist = np.abs(np.array(freq_dist) - (obs["freq_center"] / 1e6))
            min_i = np.argmin(freq_dist)
            pyfhd_config["healpix_inds"] = importlib_resources.files(
                "PyFHD.resources.healpix"
            ).joinpath(files[min_i]["name"])
        hpx_inds = load(pyfhd_config["healpix_inds"], logger=logger)
        if type(hpx_inds) is dict:
            if "nside" in hpx_inds:
                nside = hpx_inds["nside"]
            hpx_inds = hpx_inds["hpx_inds"]
    if nside is None:
        pix_sky = (
            4 * np.pi * ((180 / np.pi) ** 2) / np.prod(np.abs(obs["astr"]["cdelt"]))
        )
        nside = int(2 ** (np.ceil(np.log(np.sqrt(pix_sky / 12)) / np.log(2))))
    else:
        if not isinstance(nside, int):
            if int(nside) == nside:
                nside = int(nside)
            else:
                raise ValueError("nside must be an integer.")

    # If you wish to implement the keyword divide_pixel_area implement it here and
    # add it as an option inside pyfhd_config

    if hpx_inds is not None:
        pix_coords = np.vstack(pix2vec(nside, hpx_inds)).T
        pix_ra, pix_dec = vec2ang(pix_coords, lonlat=True)
        # This assume the refraction fix on FHD has been implemented
        xv_hpx, yv_hpx = radec_to_pixel(pix_ra, pix_dec, obs["astr"])
    else:
        cen_coords = ang2vec(obs["obsra"], obs["obsdec"], lonlat=True)
        hpx_inds = query_disc(nside, cen_coords, hpx_radius)
        pix_coords = np.vstack(pix2vec(nside, hpx_inds)).T
        pix_ra, pix_dec = vec2ang(pix_coords, lonlat=True)
        xv_hpx, yv_hpx = radec_to_pixel(pix_ra, pix_dec, obs["astr"])
        # slightly more restrictive boundary here ('LT' and 'GT' instead of 'LE' and 'GE')
        pix_i_use = np.where(
            (xv_hpx > 0)
            & (xv_hpx < obs["dimension"] - 1)
            & (yv_hpx > 0)
            & (yv_hpx < obs["elements"] - 1)
        )
        xv_hpx = xv_hpx[pix_i_use]
        yv_hpx = yv_hpx[pix_i_use]
        hpx_inds = hpx_inds[pix_i_use]
        pix_ra = pix_ra[pix_i_use]
        pix_dec = pix_dec[pix_i_use]
        if mask is not None:
            hpx_mask00 = mask[
                np.floor(xv_hpx).astype(int), np.floor(yv_hpx).astype(int)
            ]
            hpx_mask01 = mask[np.floor(xv_hpx).astype(int), np.ceil(yv_hpx).astype(int)]
            hpx_mask10 = mask[np.ceil(xv_hpx).astype(int), np.floor(yv_hpx).astype(int)]
            hpx_mask11 = mask[np.ceil(xv_hpx).astype(int), np.ceil(yv_hpx).astype(int)]
            hpx_mask = hpx_mask00 * hpx_mask01 * hpx_mask10 * hpx_mask11
            pix_i_use2 = np.nonzero(hpx_mask)
            xv_hpx = xv_hpx[pix_i_use2]
            yv_hpx = yv_hpx[pix_i_use2]
            hpx_inds = hpx_inds[pix_i_use2]
            pix_ra = pix_ra[pix_i_use2]
            pix_dec = pix_dec[pix_i_use2]

    # Test for pixels past the horizon. We don't need to be precise with this, so turn off precession, etc..
    # Get the location from obs structure
    telescope_location = EarthLocation.from_geodetic(
        lon=obs["lon"], lat=obs["lat"], height=obs["alt"]
    )
    alt, _ = radec_to_altaz(
        pix_ra,
        pix_dec,
        telescope_location.lat.value,
        telescope_location.lon.value,
        telescope_location.height.value,
        obs["jd0"],
    )
    horizon_i = np.where(alt <= 0)
    if np.size(horizon_i) > 0:
        logger.info("Cutting the HEALPix pixels that were below the horizon")
        h_use = np.where(alt > 0)
        xv_hpx = xv_hpx[h_use]
        yv_hpx = yv_hpx[h_use]
        hpx_inds = hpx_inds[h_use]
    # The differences in precision through the use of vec2ang, radec_to_pixel are exposed
    # directly causing differences in the results. We can probably assume these numbers are                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      the results. We can probably assume these numbers are
    # "better" compared to IDL
    x_frac = 1 - (xv_hpx - np.floor(xv_hpx))
    y_frac = 1 - (yv_hpx - np.floor(yv_hpx))

    v_floor = np.floor(xv_hpx) + obs["dimension"] * np.floor(yv_hpx)
    v_ceil = np.ceil(xv_hpx) + obs["dimension"] * np.ceil(yv_hpx)
    # Differences in precision occur here compared to IDL, unsolvable ones as we're using
    # built in HEALPIX and astropy functions to get these arrays. We can probably assume these
    # numbers are "better" compared to IDL, as a result the v_floor and v_ceil arrays are one off
    # compared to IDL, making min_bin off by one
    min_bin = max(np.nanmin(v_floor), 0)
    max_bin = min(np.nanmax(v_ceil), obs["dimension"] * obs["elements"] - 1)
    h00, _, ri00 = histogram(v_floor, min=min_bin, max=max_bin)
    h01, _, ri01 = histogram(
        np.floor(xv_hpx) + obs["dimension"] * np.ceil(yv_hpx), min=min_bin, max=max_bin
    )
    h10, _, ri10 = histogram(
        np.ceil(xv_hpx) + obs["dimension"] * np.floor(yv_hpx), min=min_bin, max=max_bin
    )
    h11, _, ri11 = histogram(v_ceil, min=min_bin, max=max_bin)
    htot = h00 + h01 + h10 + h11
    inds = np.nonzero(htot)[0]

    n_arr = htot[inds]
    i_use = (inds + min_bin).astype(np.int64)
    # To make sure it's an array we can store into a HDF5 file we're
    # creating object arrays full of Nones which will be populated by
    # NumPy arrays, creating a "variable length" array of arrays, to which
    # the save function in pyfhd_io can now handle through the variable_length
    # parameter
    sa = np.full(np.size(n_arr), None, dtype=object)
    ija = np.full(np.size(n_arr), None, dtype=object)

    for i in range(np.size(n_arr)):
        ind0 = inds[i]
        sa0 = np.zeros(n_arr[i])
        ija0 = np.zeros(n_arr[i], dtype=np.int64)
        hist_arr = np.array(
            [0, h00[ind0], h01[ind0], h10[ind0], h11[ind0]], dtype=np.int64
        )
        bin_i = np.cumsum(hist_arr)
        if h00[ind0] > 0:
            bi = 0
            inds1 = ri00[ri00[ind0] : ri00[ind0 + 1]]
            sa0[bin_i[bi] : bin_i[bi + 1]] = x_frac[inds1] * y_frac[inds1]
            ija0[bin_i[bi] : bin_i[bi + 1]] = inds1
        if h01[ind0] > 0:
            bi = 1
            inds1 = ri01[ri01[ind0] : ri01[ind0 + 1]]
            sa0[bin_i[bi] : bin_i[bi + 1]] = x_frac[inds1] * (1 - y_frac[inds1])
            ija0[bin_i[bi] : bin_i[bi + 1]] = inds1
        if h10[ind0] > 0:
            bi = 2
            inds1 = ri10[ri10[ind0] : ri10[ind0 + 1]]
            sa0[bin_i[bi] : bin_i[bi + 1]] = (1 - x_frac[inds1]) * y_frac[inds1]
            ija0[bin_i[bi] : bin_i[bi + 1]] = inds1
        if h11[ind0] > 0:
            bi = 3
            inds1 = ri11[ri11[ind0] : ri11[ind0 + 1]]
            sa0[bin_i[bi] : bin_i[bi + 1]] = (1 - x_frac[inds1]) * (1 - y_frac[inds1])
            ija0[bin_i[bi] : bin_i[bi + 1]] = inds1
        # Since we didn't translate pixel_area we can ignore the if statement coverring
        # the case where pixel_area is greater than 1
        sa[i] = sa0
        ija[i] = ija0

    hpx_cnv = {"nside": nside, "ija": ija, "sa": sa, "i_use": i_use, "inds": hpx_inds}
    obs["healpix"]["nside"] = nside
    if pyfhd_config["restrict_healpix_inds"]:
        obs["healpix"]["ind_list"] = None
    else:
        obs["healpix"]["ind_list"] = hpx_inds
    obs["healpix"]["n_pix"] = np.size(hpx_inds)
    mask_test = healpix_cnv_apply(mask, hpx_cnv)
    mask_test_i0 = np.where(mask_test == 0)
    obs["healpix"]["n_zero"] = np.size(mask_test_i0[0])

    return hpx_cnv, obs


def beam_image_cube(
    obs: dict,
    psf: dict | h5py.File,
    logger: Logger,
    freq_i_arr: NDArray[np.integer] | None = None,
    pol_i_arr: NDArray[np.integer] | None = None,
    n_freq_bin: int | None = None,
    square: bool = True,
    beam_threshold: float | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    Calculate the beam image per unflagged frequency channel to build a cube.
    Build a contiguous mask in x and y based off of a threshold value.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    psf : dict | h5py.File
        Beam dictionary
    logger : Logger
        PyFHD's Logger
    freq_i_arr : np.ndarray | None, optional
        Index array of the beam frequency binning in the observation
        metadata dictionary, by default None
    pol_i_arr : np.ndarray | None, optional
        Index array of the polarizations in the observation metadata
        dictionary, by default None
    n_freq_bin : int | None, optional
        Number of frequency channels to divide the full observation bandwidth
        into for the beam image cube, by default None
    square : bool, optional
        Square the beam image, by default True
    beam_threshold : float | None, optional
        Fractional threshold in which to build a mask for all smaller
        beam values, by default None

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.float64]]
        Beam image frequency cube and mask threshold array.
    """

    if beam_threshold is None:
        beam_threshold = 0.05

    if pol_i_arr is None:
        pol_i_arr = np.arange(obs["n_pol"])

    if n_freq_bin is not None:
        freq_i_arr = np.floor(
            np.arange(n_freq_bin) * (obs["n_freq"] / n_freq_bin)
        ).astype(np.int64)
    if freq_i_arr is None:
        logger.error("beam_image_cube requires n_freq_bin or freq_i_arr to be set")
        sys.exit(1)
    if n_freq_bin is None and freq_i_arr is not None:
        n_freq_bin = freq_i_arr.size

    if np.median(freq_i_arr) > obs["n_freq"]:
        freq_i_use = np.interp(
            np.arange(obs["n_freq"]), obs["baseline_info"]["freq"], freq_i_arr
        )
    else:
        freq_i_use = freq_i_arr

    beam_arr = np.zeros([obs["n_pol"], n_freq_bin, obs["dimension"], obs["elements"]])

    bin_arr = obs["baseline_info"]["fbin_i"][freq_i_use]
    bin_hist, _, bri = histogram(bin_arr, min=0)
    bin_use = np.nonzero(bin_hist)[0]
    if np.size(bin_use) == 0:
        return beam_arr
    bin_n = bin_hist[bin_use]
    beam_mask = np.ones([obs["dimension"], obs["elements"]])
    for pol_i in range(obs["n_pol"]):
        for fb_i in range(np.size(bin_use)):
            f_i_i = bri[bri[bin_use[fb_i]] : bri[bin_use[fb_i] + 1]]
            f_i = freq_i_use[f_i_i[0]]
            beam_single = beam_image(psf, obs, pol_i, freq_i=f_i, square=square)
            beam_arr[pol_i, f_i_i[0] : f_i_i[bin_n[fb_i] - 1] + 1] = beam_single
            b_i = int(obs["obsx"] + obs["obsy"] * obs["dimension"])
            beam_i = region_grow(
                beam_single,
                b_i,
                low=beam_threshold ** (square + 1),
                high=np.max(beam_single),
            )
            beam_mask1 = np.zeros([obs["dimension"], obs["elements"]])
            beam_mask1.flat[beam_i] = 1
            beam_mask *= beam_mask1
    return beam_arr, beam_mask


def phase_shift_uv_image(obs: dict) -> NDArray[np.complex128]:
    """
    Phase shift the entire u-v plane to the original phase center.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary

    Returns
    -------
    NDArray[np.float64]
        A 2D u-v phase shift array that will shift the current phase centre to
        the original phase centre.
    """
    # Since we only use it once in PyFHD, assume we always want to do /to_orig_phase
    # Implement the other options if you decide to use this function elsewhere
    ra_use = obs["orig_phasera"]
    dec_use = obs["orig_phasedec"]

    # Skip calculations if phased correctly
    if (
        obs["phasera"] == obs["orig_phasera"]
        and obs["phasedec"] == obs["orig_phasedec"]
    ):
        return np.ones([obs["dimension"], obs["elements"]], dtype=np.complex128)

    x, y = radec_to_pixel(ra_use, dec_use, obs["astr"])

    # uv_mask is not applied in FHD examples or docs decided not to translate it, if you want it put it here

    dx = (x - (obs["dimension"] / 2)) * (2 * np.pi / obs["dimension"])
    dy = (y - (obs["elements"] / 2)) * (2 * np.pi / obs["dimension"])

    xvals = meshgrid(obs["dimension"], obs["elements"], 1) - (obs["dimension"] / 2)
    yvals = meshgrid(obs["dimension"], obs["elements"], 2) - (obs["elements"] / 2)

    phase = xvals * dx + yvals * dy
    rephase_vals = np.cos(phase) + np.sin(phase) * 1j

    # Again no uv_mask therefore just return the rephase as is

    return rephase_vals


def vis_model_freq_split(
    obs: dict,
    psf: dict | h5py.File,
    params: dict,
    vis_weights: NDArray[np.float64],
    vis_model_arr: NDArray[np.complex128],
    vis_arr: NDArray[np.complex128],
    polarization: int,
    pyfhd_config: dict,
    logger: Logger,
    fft: bool = True,
    save_uvf: bool = True,
    uvf_name: str = "",
    bi_use: NDArray[np.integer] = None,
) -> dict:
    """
    Grid as a function of frequency, which can be split into larger channels. This
    uvf cube can be saved as is or used to create an orthoslant image as a function
    of frequency. Optional bi_use argument allows for the separate gridding of even/odd
    interleaved time samples for error propagation in the power spectrum.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    psf : dict | h5py.File
        Beam dictionary
    params : dict
        Visibility metadata dictionary
    vis_weights : NDArray[np.float64]
        Visibility weights array
    vis_model_arr : NDArray[np.complex128]
        Model visibility array
    vis_arr : NDArray[np.complex128]
        Calibrated visibility array
    polarization : int
        The polarization index
    pyfhd_config : dict
        PyFHD configuration settings
    logger : Logger
        PyFHD's Logger
    fft : bool, optional
        Calculate the orthoslant image instead of the u-v plane,
        by default True
    save_uvf : bool, optional
        Save the uvf cube, by default True
    uvf_name : str, optional
        The name after the obsid in the uvf cube filename, by default ''
    bi_use : NDArray[np.integer], optional
       The visibility indices to use in gridding, i.e. even/odd
       interleaved time samples, by default None

    Returns
    -------
    cube_split: dict
        Dictionary of the gridded u-v planes or orthoslant images as a function
        of frequency for the calibrated data, model (if present), residual data
        (if present), sampling map, and variance map, along with the observation
        metadata dictionary.
    """

    freq_bin_i2 = np.arange(obs["n_freq"]) // pyfhd_config["n_avg"]
    nf = int(np.max(freq_bin_i2) + 1)
    if save_uvf:
        dirty_uv_arr = np.zeros(
            [nf, obs["dimension"], obs["dimension"]], dtype=np.complex128
        )
        weights_uv_arr = np.zeros(
            [nf, obs["dimension"], obs["dimension"]], dtype=np.complex128
        )
        variance_uv_arr = np.zeros(
            [nf, obs["dimension"], obs["dimension"]], dtype=np.complex128
        )
        model_uv_arr = np.zeros(
            [nf, obs["dimension"], obs["dimension"]], dtype=np.complex128
        )
    dirty_arr = np.zeros([nf, obs["dimension"], obs["dimension"]])
    weights_arr = np.zeros([nf, obs["dimension"], obs["dimension"]])
    variance_arr = np.zeros([nf, obs["dimension"], obs["dimension"]])
    model_arr = np.zeros([nf, obs["dimension"], obs["dimension"]])
    vis_n_arr = np.zeros([nf])
    if pyfhd_config["rephase_weights"]:
        rephase_use = phase_shift_uv_image(obs)
    else:
        rephase_use = 1
    # No x_range and y_range is used in PyFHD, if you wish to do that add that here

    if bi_use is None:
        if obs["n_pol"] > 1:
            flag_test = np.maximum(np.maximum(vis_weights[0], vis_weights[1]), 0)
            # Double check the axis used
            flag_test = np.sum(flag_test, axis=1)
            bi_use = np.where(flag_test > 0)[0]
        else:
            flag_test = np.maximum(vis_weights[0], 0)
            flag_test = np.sum(flag_test, axis=1)
            bi_use = np.where(flag_test > 0)[0]

    n_vis_use = 0
    for fi in range(nf):
        fi_use = np.where((freq_bin_i2 == fi) & (obs["baseline_info"]["freq_use"] > 0))[
            0
        ]
        if np.size(fi_use) == 0:
            continue
        gridding_dict = visibility_grid(
            vis_arr[polarization],
            vis_weights[polarization],
            obs,
            psf,
            params,
            polarization,
            pyfhd_config,
            logger,
            model=vis_model_arr[polarization],
            fi_use=fi_use,
            bi_use=bi_use,
            verbose_logging=False,
        )
        if not gridding_dict:
            logger.warning(
                f"No visibilities gridded for frequency channel {fi_use} and polarization {obs['pol_names'][polarization]} ({polarization})"
            )
            continue
        n_vis_use += gridding_dict["n_vis"]
        vis_n_arr[fi] = gridding_dict["n_vis"]

        if save_uvf:
            dirty_uv_arr[fi] = gridding_dict["image_uv"] * gridding_dict["n_vis"]
            weights_uv_arr[fi] = (
                gridding_dict["weights"] * rephase_use * gridding_dict["n_vis"]
            )
            variance_uv_arr[fi] = (
                gridding_dict["variance"] * rephase_use * gridding_dict["n_vis"]
            )
            model_uv_arr[fi] = gridding_dict["model_return"] * gridding_dict["n_vis"]

        if fft:
            # No x_range and y_range hence no check for it here
            dirty_arr[fi], _, _ = dirty_image_generate(
                gridding_dict["image_uv"],
                pyfhd_config,
                logger,
                degpix=obs["degpix"],
            )
            dirty_arr[fi] *= gridding_dict["n_vis"]
            weights_arr[fi], _, _ = dirty_image_generate(
                gridding_dict["weights"] * rephase_use,
                pyfhd_config,
                logger,
                degpix=obs["degpix"],
            )
            weights_arr[fi] *= gridding_dict["n_vis"]
            variance_arr[fi], _, _ = dirty_image_generate(
                gridding_dict["variance"] * rephase_use,
                pyfhd_config,
                logger,
                degpix=obs["degpix"],
            )
            variance_arr[fi] *= gridding_dict["n_vis"]
            model_arr[fi], _, _ = dirty_image_generate(
                gridding_dict["model_return"],
                pyfhd_config,
                logger,
                degpix=obs["degpix"],
            )
            model_arr[fi] *= gridding_dict["n_vis"]
        else:
            dirty_arr[fi] = gridding_dict["image_uv"] * gridding_dict["n_vis"]
            weights_arr[fi] = (
                gridding_dict["weights"] * rephase_use * gridding_dict["n_vis"]
            )
            variance_arr[fi] = (
                gridding_dict["variance"] * rephase_use * gridding_dict["n_vis"]
            )
            model_arr[fi] = gridding_dict["model_return"] * gridding_dict["n_vis"]
    obs["n_vis"] = n_vis_use

    if save_uvf:
        h5_save_dict = {
            "dirty_uv": dirty_uv_arr,
            "weights_uv": weights_uv_arr,
            "variance_uv": variance_uv_arr,
            "model_uv": model_uv_arr,
        }
        uvf_dir = Path(Path(pyfhd_config["output_dir"], "healpix", "uvf_grid"))
        uvf_dir.mkdir(exist_ok=True, parents=True)
        save(
            Path(
                uvf_dir,
                f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_dirty_uv_arr_gridded_uvf.h5',
            ),
            h5_save_dict,
            f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_dirty_uv_arr_gridded_uvf.h5',
            logger=logger,
        )
        save(
            Path(
                uvf_dir,
                f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_weights_uv_gridded_uvf.h5',
            ),
            h5_save_dict,
            f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_weights_uv_gridded_uvf.h5',
            logger=logger,
        )
        save(
            Path(
                uvf_dir,
                f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_variance_uv_arr_gridded_uvf.h5',
            ),
            h5_save_dict,
            f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_variance_uv_arr_gridded_uvf.h5',
            logger=logger,
        )
        save(
            Path(
                uvf_dir,
                f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_model_uv_arr_gridded_uvf.h5',
            ),
            h5_save_dict,
            f'{pyfhd_config["obs_id"]}_{uvf_name}_{obs["pol_names"][polarization]}_model_uv_arr_gridded_uvf.h5',
            logger=logger,
        )

    cube_split = {
        "obs": obs,
        "residual_arr": dirty_arr,
        "weights_arr": weights_arr,
        "variance_arr": variance_arr,
        "model_arr": model_arr,
    }

    return cube_split

import copy
import logging
from math import pi, log10
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.units import Quantity

from PyFHD.pyfhd_tools.pyfhd_utils import (
    idl_argunique,
    histogram,
    angle_difference,
    parallactic_angle,
)
from PyFHD.pyfhd_tools.unit_conv import altaz_to_radec, radec_to_pixel, radec_to_altaz


def create_obs(
    pyfhd_header: dict,
    params: dict,
    layout: dict,
    pyfhd_config: dict,
    logger: logging.Logger,
) -> dict:
    """
    create_obs takes all the data that has been read in and creates the obs data structure which holds data
    and metadata of the observation we're doing a PyFHD run on. Inside this function the metafits file will
    be read as well.

    Parameters
    ----------
    pyfhd_header : dict
        The data from the UVFITS header
    params : dict
        The data from the UVFITS file
    layout : dict
        The data dictionary containing data and metadata about the antennas
    pyfhd_config : dict
        PyFHD's configuration dictionary
    logger : logging.Logger
        The PyFHD logger

    Returns
    -------
    obs : dict
        The observatiopn data structure for PyFHD containing data from the config and metadata from the observation UVFITS and METAFITS files.
    """

    obs = {}
    baseline_info = {}

    # Save the data from the header
    obs["n_pol"] = (
        pyfhd_config["n_pol"] if pyfhd_config["n_pol"] else pyfhd_header["n_pol"]
    )
    obs["n_tile"] = int(np.union1d(params["antenna1"], params["antenna2"]).size)
    obs["n_freq"] = pyfhd_header["n_freq"]
    obs["n_freq_flag"] = 0
    obs["instrument"] = pyfhd_config["instrument"]

    # keep location info for beam setup
    obs["lon"] = pyfhd_header["lon"]
    obs["lat"] = pyfhd_header["lat"]
    obs["alt"] = pyfhd_header["alt"]

    obs["obsname"] = pyfhd_config["obs_id"]
    # Deal with the times
    time = params["time"]
    b0i = idl_argunique(time)
    obs["n_time"] = b0i.size
    bin_width = np.empty(obs["n_time"])
    if obs["n_time"] > 1:
        bin_width[0 : obs["n_time"]] = b0i + 1
        b0i_range = np.arange(1, obs["n_time"])
        bin_width[b0i_range] = b0i[b0i_range] - b0i[b0i_range - 1]
    else:
        bin_width[0] = time.size
    baseline_info["bin_offset"] = np.zeros(obs["n_time"], dtype=np.int64)
    if obs["n_time"] > 1:
        baseline_info["bin_offset"][1:] = np.cumsum(bin_width[: obs["n_time"] - 1])
    # Deal with the number of visibilities
    obs["n_baselines"] = int(bin_width[0])
    obs["n_vis"] = time.size * obs["n_freq"]
    obs["n_vis_raw"] = obs["n_vis_in"] = obs["n_vis"]
    obs["nf_vis"] = np.zeros(obs["n_freq"], dtype=np.int64)

    obs["freq_res"] = pyfhd_header["freq_res"]
    baseline_info["freq"] = pyfhd_header["frequency_array"]
    if pyfhd_config["beam_nfreq_avg"] is not None:
        obs["beam_nfreq_avg"] = pyfhd_config["beam_nfreq_avg"]
    else:
        obs["beam_nfreq_avg"] = 1
    freq_bin = obs["beam_nfreq_avg"] * obs["freq_res"]
    # Let's get the freq_center and the bins we want to use
    freq_hist, _, freq_ri = histogram(baseline_info["freq"], bin_size=freq_bin)
    freq_bin_i = np.zeros(obs["n_freq"])
    for bin in range(freq_hist.size):
        if freq_ri[bin] < freq_ri[bin + 1]:
            freq_bin_i[freq_ri[freq_ri[bin] : freq_ri[bin + 1]]] = bin
    baseline_info["fbin_i"] = freq_bin_i.astype(np.int64)
    obs["freq_center"] = np.median(baseline_info["freq"])
    antenna_flag = True
    if np.max(params["antenna1"]) > 0 and (np.max(params["antenna2"]) > 0):
        baseline_info["tile_a"] = copy.deepcopy(params["antenna1"])
        baseline_info["tile_b"] = copy.deepcopy(params["antenna2"])
        antenna_flag = False
    if antenna_flag:
        # 256 tile upper limit is hard-coded in CASA format
        # these tile numbers have been verified to be correct
        # TODO: not clear that this works for more than 256 ants. e.g. LWA
        baseline_min = np.min(params["baseline_arr"])
        exponent = np.log(np.min(baseline_min)) / np.log(2)
        antenna_mod_index = 2 ** np.floor(exponent)
        tile_B_test = baseline_min % antenna_mod_index
        # Check if a bad fit and if autocorrelations or the first tile are missing
        if (tile_B_test > 1) and (baseline_min % 2 == 1):
            antenna_mod_index /= 2 ** np.floor(np.log(np.min(tile_B_test)) / np.log(2))
        baseline_info["tile_a"] = np.floor(params["baseline_arr"] / antenna_mod_index)
        baseline_info["tile_b"] = np.fix(params["baseline_arr"] / antenna_mod_index)
        if (
            max(np.max(baseline_info["tile_a"]), np.max(baseline_info["tile_b"]))
            != obs["n_tile"]
        ):
            logger.warning(
                f"Mis-matched n_tiles Header: {obs['n_tile']}, Data: {max(np.max(baseline_info['tile_a']), np.max(baseline_info['tile_b']))}, adjusting n_tiles to be same as data"
            )
            obs["n_tile"] = max(
                np.max(baseline_info["tile_a"]), np.max(baseline_info["tile_b"])
            )
        params["antenna1"] = baseline_info["tile_a"]
        params["antenna2"] = baseline_info["tile_b"]

    # check that all elements in the antenna1 and antenna2 array exist in the antenna numbers
    # from the uvfits antenna table
    all_ants = np.hstack([params["antenna1"], params["antenna2"]])
    all_ants = np.unique(all_ants)
    if not (np.all(np.isin(all_ants, layout["antenna_numbers"]))):
        logger.warning("Antenna arrays contain number(s) not found in antenna table")

    # fhd expects antenna1 and antenna2 arrays containing indices that are one-indexed.
    # Some uvfits files contain actual antenna numbers in these fields, while others
    # (particularly, those written by cotter or birli) contain indices.
    # To account for this, all antenna numbers from the uvfits header are mapped to indices
    # using the antenna numbers from the uvfits antenna table.
    # If the antenna numbers were written into the file as indices, they will be mapped to themselves.
    for tile_i in range(obs["n_tile"]):
        tile_a_antennas = np.where(
            layout["antenna_numbers"][tile_i] == params["antenna1"]
        )
        tile_b_antennas = np.where(
            layout["antenna_numbers"][tile_i] == params["antenna2"]
        )
        if np.size(tile_a_antennas) > 0:
            baseline_info["tile_a"][tile_a_antennas] = tile_i + 1
        if np.size(tile_b_antennas) > 0:
            baseline_info["tile_b"][tile_b_antennas] = tile_i + 1
    # Change the type to int to avoid issues with numba
    baseline_info["tile_a"] = baseline_info["tile_a"].astype(np.int64)
    baseline_info["tile_b"] = baseline_info["tile_b"].astype(np.int64)
    params["antenna1"] = baseline_info["tile_a"]
    params["antenna2"] = baseline_info["tile_b"]

    baseline_info["freq_use"] = np.ones(obs["n_freq"], dtype=np.int64)

    # Calculate kx and ky for each baseline at high precision to get most accurate observation information
    kx_arr = np.outer(baseline_info["freq"], params["uu"])
    ky_arr = np.outer(baseline_info["freq"], params["vv"])
    kr_arr = np.sqrt(kx_arr**2 + ky_arr**2)
    max_baseline = max(np.max(np.abs(kx_arr)), np.max(np.abs(ky_arr)))

    # Determine the imaging parameters to use
    if pyfhd_config["FoV"] is not None:
        obs["kpix"] = (180 / pi) / pyfhd_config["FoV"]
    if pyfhd_config["kbinsize"] is None:
        obs["kpix"] = 0.5
    else:
        obs["kpix"] = pyfhd_config["kbinsize"]

    # Determine observation resolution/extent parameters given number of pixels in x direction (dimension)
    if pyfhd_config["dimension"] is None and pyfhd_config["elements"] is None:
        obs["dimension"] = 2 ** int(
            (log10((2 * max_baseline) / pyfhd_config["kpix"]) / log10(2))
        )
        obs["elements"] = obs["dimension"]
    elif pyfhd_config["dimension"] is not None and pyfhd_config["elements"] is None:
        obs["dimension"] = pyfhd_config["dimension"]
        obs["elements"] = pyfhd_config["dimension"]
    elif pyfhd_config["dimension"] is None and pyfhd_config["elements"] is not None:
        obs["dimension"] = pyfhd_config["elements"]
        obs["elements"] = pyfhd_config["elements"]
    else:
        obs["dimension"] = pyfhd_config["dimension"]
        obs["elements"] = pyfhd_config["elements"]
    # Ensure both dimension and elements are ints to prevent issues down the pipeline
    obs["dimension"] = int(obs["dimension"])
    obs["elements"] = int(obs["elements"])
    obs["degpix"] = (180 / pi) / (obs["kpix"] * pyfhd_config["dimension"])

    # Set the max and min baseline
    max_baseline_inds = np.where(
        (np.abs(kx_arr) / obs["kpix"] < obs["dimension"] / 2)
        & (np.abs(ky_arr) / obs["kpix"] < obs["elements"] / 2)
    )
    obs["max_baseline"] = np.max(np.abs(kr_arr[max_baseline_inds]))
    if pyfhd_config["min_baseline"] is None:
        obs["min_baseline"] = np.min(kr_arr[np.nonzero(kr_arr)])
    else:
        obs["min_baseline"] = max(
            pyfhd_config["min_baseline"], np.min(kr_arr[np.nonzero(kr_arr)])
        )

    meta = read_metafits(obs, pyfhd_header, params, pyfhd_config, layout, logger)

    baseline_info["time_use"] = np.ones(obs["n_time"], dtype=np.int8)
    # time cut is specified in seconds to cut (rounded up to next time integration point).
    # Specify negative time_cut to cut time off the end. Specify a vector to cut at both the start and end
    if pyfhd_config["time_cut"] is not None:
        time_cut = pyfhd_config["time_cut"]
        for ti in time_cut:
            if time_cut[ti] < 0:
                ti_start = min(
                    obs["n_time"]
                    - max(np.ceil(np.abs(time_cut[ti]) / meta["time_res"]), 0),
                    obs["n_time"] - 1,
                )
                ti_end = obs["n_time"] - 1
            else:
                ti_start = 0
                ti_end = min(
                    np.ceil(np.abs(time_cut[ti])) / meta["time_res"] - 1,
                    obs["n_time"] - 1,
                )
            if ti_end >= ti_start:
                baseline_info["time_use"][ti_start : ti_end + 1] = 0
    obs["n_time_flag"] = obs["n_time"] - np.sum(baseline_info["time_use"])

    # Flag tiles based on meta data
    baseline_info["tile_use"] = 1 - meta["tile_flag"]
    obs["n_tile_flag"] = np.count_nonzero(baseline_info["tile_use"] == 0)

    # Set the last of obs values
    if pyfhd_config["dft_threshold"]:
        obs["dft_threshold"] = 1 / (2 * pi) ** 2 * obs["dimension"]
    else:
        obs["dft_threshold"] = 0
    obs["degrid_spectral_terms"] = 0
    obs["grid_spectral_terms"] = 0
    obs["alpha"] = -0.8
    obs["pol_names"] = ["XX", "YY", "XY", "YX", "I", "Q", "U", "V"]
    obs["residual"] = False

    # Setup healpix structure for obs
    healpix = {}
    healpix["nside"] = 0
    healpix["n_pix"] = 0
    # May be none!
    healpix["ind_list"] = pyfhd_config["healpix_inds"]
    healpix["n_zero"] = -1
    obs["healpix"] = healpix

    # Save the baseline_info into obs
    baseline_info["jdate"] = meta["jdate"]
    baseline_info["tile_names"] = meta["tile_names"]
    baseline_info["tile_height"] = meta["tile_height"]
    baseline_info["tile_flag"] = meta["tile_flag"]
    obs["baseline_info"] = baseline_info

    # Save the last of the metadata into obs
    for key in meta.keys():
        if key not in baseline_info.keys():
            obs[key] = meta[key]

    return obs


def read_metafits(
    obs: dict,
    pyfhd_header: dict,
    params: dict,
    pyfhd_config: dict,
    layout: dict,
    logger: logging.Logger,
) -> dict:
    """
    Reads the metafits file provided inside the same input directory as the UVFITS file.
    It will process the data found in the METAFITS file and then returns a meta dictionary.
    Which will eventually end up inside the obs dictionary.

    Parameters
    ----------
    obs : dict
        The current obs structure without the metadata
    pyfhd_header : dict
        The data from the UVFITS header
    params : dict
        The data from the UVFITS file
    pyfhd_config : dict
        PyFHD's configuration dictionary
    layout : dict
        Info from the uvfits antenna table
    logger : logging.Logger
        PyFHD's logger

    Returns
    -------
    meta : dict
        The dictionary holding the metadata from the UVFITS and METAFITS files
    """

    meta = {}
    time = params["time"]
    b0i = idl_argunique(time)
    meta["jdate"] = time[
        b0i
    ]  # Time is already in julian. No need to add the bzero (or pzero) value
    meta["obsx"] = obs["dimension"] // 2
    meta["obsy"] = obs["elements"] // 2
    meta["jd0"] = np.min(meta["jdate"])
    meta["epoch"] = Time(meta["jd0"], format="jd").to_value("decimalyear")
    meta_path = Path(pyfhd_config["input_path"], pyfhd_config["obs_id"] + ".metafits")
    if meta_path.is_file():
        metadata = fits.open(meta_path)
        hdr = metadata[0].header
        data = metadata[1].data
        # Sort the data by antenna using a stable sort, astropy Table is required to access Antenna column for sorting
        # Standard Astropy does not do stable sorting, hence use of argsort to do stable sorting
        data = data[np.array(Table(data).argsort("Antenna", kind="stable"))]
        single_i = np.where(data["pol"] == data["pol"][0])
        meta["tile_names"] = data["tile"][single_i]
        meta["tile_height"] = data["height"][single_i] - pyfhd_header["alt"]
        meta["tile_flag"] = data["flag"][single_i]
        if np.sum(meta["tile_flag"]) == meta["tile_flag"].size - 1:
            if pyfhd_config["run_simulation"]:
                logger.warning("All tiles flagged in metadata")
            else:
                logger.error("All tiles flagged in metadata")
                exit()
        meta["obsra"] = hdr["RA"]
        meta["obsdec"] = hdr["DEC"]
        meta["phasera"] = hdr["RAPHASE"]
        meta["phasedec"] = hdr["DECPHASE"]
        meta["time_res"] = hdr["INTTIME"]
        delays = hdr["DELAYS"].split(",")
        meta["delays"] = (
            np.asarray(delays, np.int64)
            .repeat(obs["n_pol"])
            .reshape((obs["n_pol"], len(delays)))
        )
    else:
        logger.warning(
            "METAFITS file has not been found, Calculating obs meta settings from the uvfits header instead"
        )
        hdr = None
        # Simulate the flagging of tiles by taking where tiles don't exist
        tile_A1 = params["antenna1"]
        tile_B1 = params["antenna2"]
        hist_A1, _, _ = histogram(tile_A1, min=1, max=obs["n_tile"])
        hist_B1, _, _ = histogram(tile_B1, min=1, max=obs["n_tile"])
        hist_AB = hist_A1 + hist_B1
        meta["tile_names"] = layout["antenna_names"]
        # if pyuvdata is available, get antenna heights using pyuvdata utils
        try:
            from pyuvdata import utils

            telescope_location = EarthLocation.from_geodetic(
                lon=obs["lon"], lat=obs["lat"], height=obs["alt"]
            )
            ant_ecef = utils.ECEF_from_rotECEF(
                layout["antenna_coords"], longitude=telescope_location.lon.rad
            )
            # in uvfits the antenna positions are relative to the array center.
            # need to add those back in for the next calc
            ant_ecef += Quantity(telescope_location.geocentric).to_value("m")
            ant_enu = utils.ENU_from_ECEF(ant_ecef, center_loc=telescope_location)
            meta["tile_height"] = ant_enu[:, 2]
        except ImportError:
            meta["tile_height"] = np.zeros(obs["n_tile"])
        tile_use = np.where(hist_AB == 0)[0]
        meta["tile_flag"] = np.zeros(obs["n_tile"], dtype=np.int8)
        if tile_use.size > 0:
            meta["tile_flag"][tile_use] = 1
        if b0i.size > 1:
            meta["time_res"] = (time[b0i[1]] - time[b0i[0]]) * 24.0 * 3600.0
        else:
            meta["time_res"] = 1
        meta["obsra"] = pyfhd_header["obsra"]
        meta["obsdec"] = pyfhd_header["obsdec"]
        meta["phasera"] = pyfhd_header["obsra"]
        meta["phasedec"] = pyfhd_header["obsdec"]
        meta["delays"] = None

    # Store an origin/target phase ra/dec
    meta["orig_phasera"] = (
        pyfhd_config["override_target_phasera"]
        if pyfhd_config["override_target_phasera"] is not None
        else meta["phasera"]
    )
    meta["orig_phasedec"] = (
        pyfhd_config["override_target_phasedec"]
        if pyfhd_config["override_target_phasedec"] is not None
        else meta["phasedec"]
    )

    # Get the Zenith RA and DEC from the location and time
    zenra, zendec = altaz_to_radec(
        90,
        0,
        pyfhd_header["lat"],
        pyfhd_header["lon"],
        pyfhd_header["alt"],
        meta["jd0"],
    )
    meta["zenra"] = zenra
    meta["zendec"] = zendec

    # Project Slant Orthographic
    # astr is basically a WCS data structure.
    # zenx and zeny are the Zenith pixel coordinates
    meta["astr"], meta["zenx"], meta["zeny"] = project_slant_orthographic(meta, obs)

    # Get the alt and azimuth of the observation
    meta["obsalt"], meta["obsaz"] = radec_to_altaz(
        meta["obsra"],
        meta["obsdec"],
        pyfhd_header["lat"],
        pyfhd_header["lon"],
        pyfhd_header["alt"],
        meta["jd0"],
    )

    if hdr is not None:
        # Save the raw header and data into the meta dictionary
        # Save the header as a Python dictionary
        meta["meta_hdr"] = {}
        for key in hdr.keys():
            # Check if they is HISTORY or COMMENT which will be changed to a list for ease of use with hdf5 files
            if key in ["HISTORY", "COMMENT"]:
                meta["meta_hdr"][key] = list(hdr[key])
            else:
                meta["meta_hdr"][key] = hdr[key]
        # The astropy FITS_rec class is based off a numpy record array so saving as is should be fine
        # If so desired a tolist() function to turn the data into list of lists, but you lose the column names
        meta["meta_data"] = data

    return meta


def project_slant_orthographic(meta: dict, obs: dict, epoch=2000) -> dict:
    """
    Create an astrometry data structure holding key astrometry information.
    It's essentially a WCS data structure, done as a Python dictionary allowing greater compatibility
    with other packages.

    Parameters
    ----------
    meta : dict
        The current metadata dictionary
    obs : dict
        The current obs dictionary
    epoch : float
        The equinox used for the dictionary structure

    Returns
    -------
    astr : dict
        An astrometry structure built from meta and obs
    """

    if abs(meta["phasera"] - meta["zenra"]) > 90:
        lon_offset = (
            meta["phasera"]
            - (360 if meta["phasera"] > meta["zenra"] else -360)
            - meta["zenra"]
        )
    else:
        lon_offset = meta["phasera"] - meta["zenra"]
    zenith_ang = angle_difference(
        meta["phasera"], meta["phasedec"], meta["zenra"], meta["zendec"], degree=True
    )
    parallactic_ang = parallactic_angle(meta["zendec"], lon_offset, meta["phasedec"])

    xi = -1 * np.tan(np.radians(zenith_ang)) * np.sin(np.radians(parallactic_ang))
    eta = np.tan(np.radians(zenith_ang)) * np.cos(np.radians(parallactic_ang))

    # Replicate MAKE_ASTR return dictionary structure from astrolib
    # We don't have to do it perfectly as it's only used for this function with the above as inputs
    # This is essentially a WCS in a dictionary for use with other libraries other than Astropy
    astr = {}
    astr["naxis"] = np.array([obs["dimension"], obs["elements"]])
    astr["cd"] = np.identity(2)
    astr["cdelt"] = np.full(2, obs["degpix"])
    astr["crpix"] = np.array([meta["obsx"], meta["obsy"]]) + 1
    astr["crval"] = np.array([meta["phasera"], meta["phasedec"]])
    projection_name = "SIN"
    astr["ctype"] = ["RA---" + projection_name, "DEC--" + projection_name]
    astr["longpole"] = 180
    astr["latpole"] = 0
    astr["pv2"] = np.array([xi, eta])
    # The PV1 array in Astrolib ASTR contains 5 projection parameters associated with longitude axis
    # [xyoff, phi0, theta0, longpole, latpole]
    # xyoff and phi0 are 0 as default
    # The third number [i = 2] is determined by the fact we are using SIN zenithal projections
    # The last are the longpole and latpole we set earlier
    astr["pv1"] = np.array([0, 0, 90, 180, 0], dtype=np.float64)
    astr["axes"] = np.array([1, 2])
    astr["reverse"] = (
        0  # Since Axes are always valid Celestial, we don't need to reverse them
    )
    astr["coord_sys"] = "C"  # Celestial Coordinate System in MAKE_ASTR
    astr["projection"] = projection_name
    astr["known"] = np.array([1])  # The projection name is guaranteed to be known
    astr["radecsys"] = "ICRS"  # Using ICRS instead of FK5
    astr["equinox"] = epoch
    astr["dateobs"] = Time(meta["jd0"], format="jd").to_value("fits")
    astr["mjdobs"] = meta["jd0"] - 2400000.5
    astr["x0y0"] = np.zeros(2, dtype=np.float64)
    # Get the pixel coordinates of zenra and zendec
    zenx, zeny = radec_to_pixel(meta["zenra"], meta["zendec"], astr)

    return astr, zenx, zeny


def update_obs(
    obs: dict,
    dimension: int,
    kbinsize: float | int,
    beam_nfreq_avg: float | None = None,
    fov: float | None = None,
) -> dict:
    """
    Inside the quickview function for exporting files we need to update the obs
    dictionary based on the new dimension and kbinsize given. This differs slightly from
    FHD as we only adjust the exact things required for this as we only use this function
    once in quickview.

    Parameters
    ----------
    obs : dict
        The original observation dictionary
    dimension : int
        The new dimension for the size of each axes
    kbinsize : float | int
        The new kbin
    beam_nfreq_avg: float | None
        Set the new factor to average up the frequency resolution,by default None
    fov: float | None
        Set a new field of view, by default None

    Returns
    -------
    obs: dict
        The new updated obs dictionary
    """
    if beam_nfreq_avg is None:
        beam_nfreq_avg = np.round(
            obs["n_freq"] / (np.max(obs["baseline_info"]["fbin_i"]) + 1)
        )
    freq_bin = beam_nfreq_avg * obs["freq_res"]
    freq_hist, _, freq_ri = histogram(obs["baseline_info"]["freq"], bin_size=freq_bin)
    freq_bin_i = np.zeros(obs["n_freq"], dtype=np.int64)
    for bin in range(freq_hist.size - 1):
        if freq_ri[bin] < freq_ri[bin + 1]:
            freq_bin_i[freq_ri[freq_ri[bin] : freq_ri[bin + 1]]] = bin
    # Adjust the obs dictionary based on the new dimension and kbinsize
    obs["dimension"] = dimension
    obs["elements"] = dimension
    obs["obsx"] = dimension / 2
    obs["obsy"] = dimension / 2
    obs["kpix"] = kbinsize if fov is None else (180 / np.pi) / fov
    obs["degpix"] = (180 / np.pi) / (obs["kpix"] * dimension)
    obs["max_baseline"] = min(
        obs["max_baseline"], (dimension * obs["kpix"]) / np.sqrt(2)
    )
    obs["astr"]["naxis"] = np.array([dimension, dimension])
    obs["astr"]["cdelt"] = np.full(2, obs["degpix"])
    obs["astr"]["crpix"] = np.array([obs["obsx"] + 1, obs["obsy"] + 1])
    obs["baseline_info"]["fbin_i"] = freq_bin_i

    return obs

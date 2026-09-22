import importlib_resources
import logging
from typing import Literal

import numpy as np
from astropy.constants import c
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units
from astropy.time import Time
from pyuvdata import BeamInterface, UVBeam
from scipy.interpolate import interp1d

from ..pyfhd_tools.unit_conv import pixel_to_radec, radec_to_altaz
from ..pyfhd_tools.pyfhd_utils import parallactic_angle
from ..pyfhd_tools.types import ComplexArray, FloatArray


def _azza_beam_arrays(image_dim, scale, obs, jdate):
    # Get the zenith angle and azimuth angle arrays

    location = EarthLocation.from_geodetic(
        lon=obs["lon"], lat=obs["lat"], height=obs["alt"]
    )

    xvals_celestial, yvals_celestial = np.meshgrid(
        np.arange(image_dim), np.arange(image_dim)
    )
    xvals_celestial = xvals_celestial * scale - image_dim * scale / 2 + obs["obsx"]
    yvals_celestial = yvals_celestial * scale - image_dim * scale / 2 + obs["obsy"]
    ra_arr, dec_arr = pixel_to_radec(xvals_celestial, yvals_celestial, obs["astr"])
    del xvals_celestial, yvals_celestial
    # Only keep the pixels that are above the horizon to save memory
    valid_i = np.nonzero(np.isfinite(ra_arr))
    ra_arr = ra_arr[valid_i]
    dec_arr = dec_arr[valid_i]
    alt_arr, az_arr = radec_to_altaz(
        ra=ra_arr.value,
        dec=dec_arr.value,
        lat=location.lat.value,
        lon=location.lon.value,
        height=location.height.value,
        time=jdate,
    )
    # astropy's WCS is being used to go from x/y grid values to RA/Dec
    # then using SkyCoord to go from RA/Dec to alt/az
    # the first step should drop pixels beyond the horizon, but occasionally a
    # few pixels survive that step but have negative altitudes after the second
    # step which breaks things downstream. drop those pixels.
    # TODO: understand this apparent disagreement between WCS & SkyCoord...
    good_alt = np.nonzero(alt_arr >= 0.0)[0]

    # valid_i is a tuple with per-dimension indices.
    valid_i = list(valid_i)
    for ind, elem in enumerate(valid_i):
        valid_i[ind] = elem[good_alt]
    valid_i = tuple(valid_i)

    alt_arr = alt_arr[good_alt]
    az_arr = az_arr[good_alt]
    ra_arr = ra_arr[good_alt]
    dec_arr = dec_arr[good_alt]

    # Convert to radians for pyuvdata functions
    return ra_arr, dec_arr, np.deg2rad(90 - alt_arr), np.deg2rad(az_arr), valid_i


def jones_to_mueller(jones: FloatArray | ComplexArray) -> FloatArray | ComplexArray:
    """
    Calculate a Mueller matrix from a Jones matrix.

    This is a re-ordered Kronecker product of the Jones matrix with its complex
    conjugate. The Kronecker product is only done over the first 2 dimensions,
    which are required to both be length 2 and the result is reordered along the
    first 2 dimensions to give the expected ordering (the coherency axis (0th axis)
    will be e.g. RR, DD, RD, DR for RA/Dec; the pol axis (1st axis) will be
    xx, yy, xy, yx).
    The other dimensions (e.g. frequency, direction on the sky) should just carry
    through. We don't use the numpy `kron` function because it doesn't allow for
    restricting the dimensions.

    Parameters
    ----------
    jones : np.ndarray
        Jones or similar (e.g. feed aligned projection) matrix. First 2 dimensions
        must be length 2.

    Returns
    -------
    mueller : np.ndarray
        Kronecker product of input matrix over the first 2 dimensions. First 2
        dimensions will be length 4.

    """
    if jones.ndim < 2 or jones.shape[0] != 2 or jones.shape[1] != 2:
        raise ValueError("Input jones matrix must have first 2 dimensions be length 2.")

    # Compute the product
    a_arr = np.expand_dims(jones, axis=tuple(range(1, 4, 2)))
    b_arr = np.expand_dims(np.conjugate(jones), axis=tuple(range(0, 4, 2)))
    mueller = np.multiply(a_arr, b_arr)

    mueller_shape = tuple([4, 4] + list(jones.shape[2:]))
    mueller = mueller.reshape(mueller_shape)

    # reorder axes
    sort_order = np.array([0, 3, 1, 2])
    mueller = mueller[sort_order]
    mueller = mueller[:, sort_order]

    return mueller


def get_beam_values(
    beam_obj: BeamInterface,
    za_array: FloatArray | None = None,
    alt_array: FloatArray | None = None,
    az_array: FloatArray | None = None,
    ra_array: FloatArray | None = None,
    dec_array: FloatArray | None = None,
    az_convention: Literal["east of north", "north of east"] = "east of north",
    frame: str = "fk5",
    time: Time | None = None,
    telescope_location: EarthLocation | None = None,
    freq_array: FloatArray | None = None,
    spline_opts: dict | None = None,
    check_azza_domain: bool = True,
) -> ComplexArray:
    """
    Get beam values from a pyuvdata beam for a set of directions on the sky.

    Accepts zenith angle and azimuth, altitude and aziumth or RA/Dec arrays
    along with the associated frame and astropy Time and EarthLocation objects.
    Azimuth convention is specified using the `az_convention` parameter,
    options are "north of east" (the UVBeam convention) or "east of north"
    (the astropy alt/az frame convention and the FHD convention).

    Parameters
    ----------
    beam_obj : BeamInterface
        A pyuvdata BeamInterface object.
    alt_array : ndarray of float
        Array of altitudes (also called elevations) in radians. Must be a 1D array.
    za_array : ndarray of float
        Array of zenith angles (zenith is zero, horizon is 90 degrees). Must be
        a 1D array.
    az_array : ndarray of float
        Array of azimuths in radians. Defined according to the az_convention parameter.
        Must be a 1D array.
    ra_array : ndarray of float
        Array of right ascensions in radians. Must be a 1D array.
    dec_array : ndarray of float
        Array of declinations in radians. Must be a 1D array.
    az_convention : str
        either "east of north" N=0, E=90 degrees or "north of east" E=0, N=90 degrees.
    frame : str
        The frame for RA and Dec, ignored if alt/az are provided. Must be a frame
        known to astropy. Defaults to FK5.
    time : astropy.time.Time
        Astropy Time object specifying the center of the observation time. Used
        for converting RA/Dec to AltAz, ignored if alt/az are provided.
    telescope_location : astropy.coordinates.EarthLocation
        Astropy EarthLocation object specifying the telescope location. Used
        for converting RA/Dec to AltAz, ignored if alt/az are provided.
    freq_array : ndarray of float
        Frequencies to get the beam response for in Hz. Requried for analytic beams,
        defaults to the frequencies defined on the beam object for UVBeams.
    spline_opts : dict
        Provide options to numpy.RectBivariateSpline. This includes spline
        order parameters `kx` and `ky`, and smoothing parameter `s`. Only
        applies if beam is a UVBeam.
    check_azza_domain : bool
        Whether to check the domain of az/za to ensure that they are covered by the
        intrinsic data array. Checking them can be quite computationally expensive.
        Conversely, if the passed az/za are outside of the domain, they will be
        silently extrapolated and the behavior is not well-defined. Only
        applies if beam is a UVBeam. Should be set to False if it is known that
        the beam covers the whole sky.

    Returns
    -------
    beam_vals : ndarray of complex
        The beam values at the requested locations.

    """
    alt_az_in = alt_array is not None and az_array is not None
    za_az_in = za_array is not None and az_array is not None
    ra_dec_in = np.all(
        [var is not None for var in [ra_array, dec_array, time, telescope_location]]
    )

    if not alt_az_in and not za_az_in and not ra_dec_in:
        raise ValueError(
            "Either alt_array and az_array must be provided or ra_array, dec_array, "
            "time and telescope_location must all be provided."
        )

    allowed_az_convention = ["east of north", "north of east"]
    if (alt_az_in or za_az_in) and (az_convention not in allowed_az_convention):
        raise ValueError(
            f"az_convention must be one of {allowed_az_convention}. "
            f"It was {az_convention}."
        )

    if ra_dec_in:
        if ra_array.shape != dec_array.shape:
            raise ValueError("ra_array and dec_array must have the same shape")

        # convert to alt/az
        skycoord = SkyCoord(
            ra=ra_array * units.rad,
            dec=dec_array * units.rad,
            frame=frame,
            obstime=time,
            location=telescope_location,
        ).transform_to("altaz")

        alt_array = skycoord.alt.to("rad").value
        az_array = skycoord.az.to("rad").value
        az_convention = "east of north"
    elif alt_az_in:
        if alt_array.shape != az_array.shape:
            raise ValueError("alt_array and az_array must have the same shape")
    else:
        if za_array.shape != az_array.shape:
            raise ValueError("za_array and az_array must have the same shape")

    if alt_az_in or ra_dec_in:
        za_array = np.pi / 2 - alt_array

    if az_convention == "east of north":
        noe_az_array = np.pi / 2 - az_array
    else:
        noe_az_array = az_array

    # Wrap the azimuth array to [0, 2pi] to match the extent of the UVBeam azimuth
    where_neg_az = np.nonzero(noe_az_array < 0)
    noe_az_array[where_neg_az] = noe_az_array[where_neg_az] + np.pi * 2.0

    beam_vals = beam_obj.compute_response(
        az_array=noe_az_array,
        za_array=za_array,
        freq_array=freq_array,
        spline_opts=spline_opts,
        check_azza_domain=check_azza_domain,
    )

    return beam_vals


logger = logging.getLogger(__name__)


def init_beam(obs: dict, pyfhd_config: dict) -> dict:
    """
    Build an antenna-specific metadata dictionary and a full station/tile power
    beam model and dictionary. Currently, the jones matrix of the antenna is
    acquired through pyuvdata.

    The antenna dictionary contains antenna parameters as well as response and
    jones matrices that can be used to build a beam power response. The psf
    dictionary contains parameters and options required to build a uv-response
    from that beam power with reduced aliasing contamination.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary.
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options for a run

    Returns
    -------
    antenna : dict
        Antenna metadata dictionary, including feed-aligned matrices.
    psf : dict
        Beam metadata dictionary for the station/tile.
    beam : UVBeam or AnalyticBeam
        A pyuvdata beam, can be a UVBeam, and AnalyticBeam subclass, or a
        BeamInterface object.

    Raises
    ------
    FileNotFoundError
        If the MWA beam file does not exist and needs to be downloaded.
    ValueError
        If the instrument is not supported or if the antenna configuration is invalid.
    """

    # Setup the constants and variables
    # Almost all instruments have two instrumental polarizations (either linear
    # or circular)
    n_ant_pol = 2
    frequency_array = obs["baseline_info"]["freq"]
    freq_bin_i = obs["baseline_info"]["fbin_i"]
    nfreq_bin = int(np.max(freq_bin_i)) + 1
    tile_a = obs["baseline_info"]["tile_a"]
    tile_b = obs["baseline_info"]["tile_b"]
    ant_names = np.unique(
        np.concatenate((tile_a[: obs["n_baselines"]], tile_b[: obs["n_baselines"]]))
    )
    if pyfhd_config["beam_offset_time"] is not None:
        jdate_use = obs["jd0"] + pyfhd_config["beam_offset_time"] / 24 / 3600
    else:
        jdate_use = obs["jd0"]

    freq_center = np.zeros(nfreq_bin)
    interp_func = interp1d(freq_bin_i, frequency_array)
    for fi in range(nfreq_bin):
        fi_i = np.where(freq_bin_i == fi)[0]
        if fi_i.size == 0:
            freq_center[fi] = interp_func(fi)
        else:
            freq_center[fi] = np.median(frequency_array[fi_i])

    antenna_size = {"mwa": 5, "hera": 14}
    if pyfhd_config["instrument"] in antenna_size:
        ant_size_m = antenna_size[pyfhd_config["instrument"]]
    else:
        ant_size_m = 10

    if pyfhd_config["instrument"] == "mwa":
        # Get the antenna coordinates
        n_dipoles = 16
        antenna_spacing = 1.1
        xc_arr, yc_arr = np.meshgrid(np.arange(4), np.arange(4))
        xc_arr = xc_arr.flatten() * antenna_spacing
        yc_arr = np.flipud(yc_arr).flatten() * antenna_spacing
        zc_arr = np.zeros(n_dipoles)
        coords = np.array([xc_arr, yc_arr, zc_arr])
        # Get the delays
        delays = obs["delays"] * 4.35e-10
    else:
        n_dipoles = 1
        coords = np.zeros((3, n_dipoles))
        delays = np.zeros(n_dipoles)

    # Create basic antenna dictionary
    antenna = {
        "n_pol": n_ant_pol,
        "antenna_type": pyfhd_config["instrument"],
        "size_meters": ant_size_m,
        "names": ant_names,
        "freq": freq_center,
        "nfreq_bin": nfreq_bin,
        "n_ant_elements": n_dipoles,
        # Anything that was pointer arrays in IDL will be None until assigned in Python
        "gain": np.ones([n_ant_pol, freq_center.size, n_dipoles], dtype=np.float64),
        "coords": coords,
        "delays": delays,
        # The aligned response is needed on the psf["image_dim"] size for the
        # gridding kernel
        "aligned_response_psf_image": None,
        # The L matrix (kronecker product of the K -- the aligned projection) is
        # needed on the obs["dimension"] size for stokes_cnv
        "l_matrix_image_radec": None,
        # Also need L inverse
        "l_inv_image_radec": None,
        # pyfhd supports one antenna beam at a time, so we setup the group so
        # all the baselines are in the same group.
        "group_id": np.zeros([n_ant_pol, obs["n_tile"]], dtype=np.int8),
        "pix_window": None,
    }

    # psf_dim is *required* to be even
    if pyfhd_config["psf_dim"] is not None:
        psf_dim = pyfhd_config["psf_dim"]
    else:
        psf_dim = np.ceil(
            antenna["size_meters"]
            * 2
            * np.max(obs["baseline_info"]["freq"])
            / (c.value * obs["kpix"])
        )
        psf_dim = int(np.ceil(psf_dim / 2) * 2)

    # Create the initial psf dict
    psf = {
        "dim": psf_dim,
        "resolution": pyfhd_config["psf_resolution"],
        # This is more of a placeholder, if we want pyfhd to support processing
        # more than one instrument at a time we'll need to edit this to be
        # calculated rather than hardcoded.
        "id": np.zeros(
            [obs["n_pol"], obs["n_freq"], obs["n_baselines"]], dtype=np.int64
        ),
        "beam_mask_threshold": pyfhd_config["beam_mask_threshold"],
        "freq_norm": np.ones(obs["n_freq"], dtype=np.int64),
        "image_resolution": 10,  # Add psf_image_resolution to the config file
        "fbin_i": obs["baseline_info"]["fbin_i"],
        "interpolate_kernel": pyfhd_config["interpolate_kernel"],
    }

    # Set up the beam objects we will need
    if pyfhd_config["analytic_beam_yaml"] is not None:
        beam = pyfhd_config["analytic_beam_yaml"]
        f_beam = BeamInterface(beam, beam_type="feed_aligned_response")
        k_beam = BeamInterface(beam, beam_type="feed_aligned_projection")
        k_freq_array = np.mean(freq_center)
    else:
        # only read in the range of beam frequencies we need. add a buffer
        # to ensure that we have enough outside our range for interpolation
        uvbeam_kwargs = {}
        if pyfhd_config["instrument"] == "mwa":
            uvbeam_kwargs["delays"] = obs["delays"]

        if pyfhd_config["uvbeam_zfile_path"] is not None:
            uvbeam_kwargs["mwa_zfile"] = pyfhd_config["uvbeam_zfile_path"]

        if pyfhd_config["uvbeam_freq_buffer"] is not None:
            freq_range = [
                np.min(obs["baseline_info"]["freq"])
                - pyfhd_config["uvbeam_freq_buffer"],
                np.max(obs["baseline_info"]["freq"])
                + pyfhd_config["uvbeam_freq_buffer"],
            ]
            uvbeam_kwargs["freq_range"] = freq_range

        uvbeam_kwargs["mwa_include_cross_feed_coupling"] = pyfhd_config[
            "uvbeam_mwa_include_cross_feed_coupling"
        ]

        # select above the horizon (this only does something for beamfits files)
        # but it saves some memory in that case
        uvbeam_kwargs["za_range"] = [0, 90.0]

        if pyfhd_config["uvbeam_file_path"] is not None:
            beam = UVBeam.from_file(pyfhd_config["uvbeam_file_path"], **uvbeam_kwargs)
        elif pyfhd_config["instrument"] == "mwa":
            # fall back to looking for the MWA beam in resources folder (older pattern)
            mwa_beam_file = importlib_resources.files(
                "pyfhd.resources.instrument_config"
            ).joinpath("mwa_full_embedded_element_pattern.h5")
            if not mwa_beam_file.exists():
                # Download the MWA beam file if it does not exist
                raise FileNotFoundError(
                    f"MWA beam file {mwa_beam_file} does not exist. "
                    "Please download it from "
                    "http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5"
                    f" into the directory {mwa_beam_file.parent}"
                )
            beam = UVBeam.from_file(mwa_beam_file, **uvbeam_kwargs)

        # check for nans in beam. If they can be removed by a horizon cut do it.
        if np.any(np.isnan(beam.data_array)):
            if beam.pixel_coordinate_system != "healpix":
                above_hor_inc = np.nonzero(beam.axis2_array > (np.pi / 2))[0]
                above_hor_exc = np.nonzero(beam.axis2_array >= (np.pi / 2))[0]
                if not np.any(np.isnan(beam.data_array[above_hor_inc])):
                    logger.info("Cutting the beam below the horizon to remove NaNs.")
                    beam.select(axis2_inds=above_hor_inc)
                elif not np.any(np.isnan(beam.data_array[above_hor_exc])):
                    logger.info(
                        "Cutting the beam at and below the horizon to remove NaNs."
                    )
                    beam.select(axis2_inds=above_hor_exc)
                else:
                    raise ValueError(
                        "UVBeam object has NaNs in the data array above the horizon."
                    )
        f_obj, k_obj = beam.decompose_feed_aligned_terms()
        f_beam = BeamInterface(f_obj)

        # Need to have a single k matrix across all frequencies. It is a research
        # question how to do that. Options include:
        # - average after decomposition
        # - average before decomposition
        # - use the middle frequency
        # - use K from an analytic dipole
        # for now, averaging after decomposition (so averaging Ks)
        # some quick testing suggested this gave smoother results than averaging
        # before decomposition. It also seems like it would give smoother results
        # because the Ks are more similar across frequency than the Js are.
        ave_k_beam = k_obj.copy()
        ave_k_beam.select(freq_chans=[0])
        ave_k_beam.freq_array = np.atleast_1d(np.mean(k_obj.freq_array))
        ave_k_beam.data_array[:, :, 0] = np.mean(ave_k_beam.data_array, axis=2)
        k_freq_array = ave_k_beam.freq_array
        k_beam = BeamInterface(ave_k_beam)

    # We need beam products on two different image sizes: the obs["dimension"]
    # size and the psf["image_dim"] size. We need the feed-aligned response (F)
    # on the psf["image_dim"] size for the gridding kernel and the feed-aligned
    # projection (K) on the obs["dimension"] size for converting between instrumental
    # pol and Stokes (in stokes_cnv).

    # Set up coordinates for the gridding kernel to generate the high uv
    # resolution model.
    # Remember that field of view = uv resolution, image pixel scale = uv span.
    # So, the cropped uv span (psf_dim) means we do not need to calculate at
    # full image resolution, while the increased uv resolution can correspond to
    # super-horizon scales. We construct the beam model in image space, and while
    # we don't need the full image resolution we need to avoid quantization
    # errors that come in if we make too small an image and then take the FFT
    psf["intermediate_res"] = np.min(
        [np.ceil(np.sqrt(psf["resolution"]) / 2) * 2, psf["resolution"]]
    ).astype(int)
    # use a larger box to build the model than will ultimately be used, to
    # allow higher resolution in the initial image space beam model
    psf["image_dim"] = int(
        psf["dim"] * psf["image_resolution"] * psf["intermediate_res"]
    )
    psf["scale"] = obs["dimension"] * psf["intermediate_res"] / psf["image_dim"]
    psf["pix_horizon"] = obs["dimension"] / psf["scale"]
    psf["superres_dim"] = psf["dim"] * psf["resolution"]

    _, _, psf_zenith_angle_arr, psf_azimuth_arr, psf_inds = _azza_beam_arrays(
        image_dim=psf["image_dim"], scale=psf["scale"], obs=obs, jdate=jdate_use
    )

    # Store pixel indices above the horizon
    antenna["psf_image_pix_use"] = np.ravel_multi_index(
        psf_inds, (psf["image_dim"], psf["image_dim"])
    )

    # First get the feed aligned response (F) on the psf["image_dim"] scale
    # remove the initial shallow dimension in aligned_response
    # don't use squeeze in case only 1 freq.
    antenna["aligned_response_psf_image"] = get_beam_values(
        f_beam,
        za_array=psf_zenith_angle_arr.flatten(),
        az_array=psf_azimuth_arr.flatten(),
        freq_array=freq_center,
    )[0]

    # Next get the feed aligned projection (K) on the obs["dimension"] scale
    (
        image_ra_arr,
        image_dec_arr,
        image_zenith_angle_arr,
        image_azimuth_arr,
        image_inds,
    ) = _azza_beam_arrays(image_dim=obs["dimension"], scale=1, obs=obs, jdate=jdate_use)
    # Store pixel indices above the horizon
    antenna["image_pix_use"] = np.ravel_multi_index(
        image_inds, (obs["dimension"], obs["dimension"])
    )

    # use squeeze to remove shallow freq axis
    k_vals_image = get_beam_values(
        k_beam,
        za_array=image_zenith_angle_arr.flatten(),
        az_array=image_azimuth_arr.flatten(),
        freq_array=k_freq_array,
    ).squeeze()

    # need to rotate aligned projection input from Alt/Az to RA/Dec using the
    # parallactic angle to get a projection matrix to go between Stokes and
    # instrumental pols.

    hour_angle = obs["zenra"] - image_ra_arr.value
    hour_angle[np.nonzero(hour_angle < 0)] += 360
    hour_angle[np.nonzero(hour_angle > 360)] -= 360

    # Calculate the parallactic angle.
    # Parallactic angle is the angle between the -ZA axis and the +Dec axis.
    # A rotation angle of 0 degrees is defined with
    # North along the -ZA axis and East along the -Az axis.
    # A rotation angle of 90 degrees is defined with
    # North along the +Az axis and East along the -ZA axis.
    par_ang = parallactic_angle(
        latitude=obs["zendec"], hour_angle=hour_angle, dec=image_dec_arr
    )

    par_rot_matrix = np.zeros_like(k_vals_image)
    par_rot_matrix[0, 0] = np.sin(np.deg2rad(par_ang))
    par_rot_matrix[1, 0] = -np.cos(np.deg2rad(par_ang))
    par_rot_matrix[0, 1] = -np.cos(np.deg2rad(par_ang))
    par_rot_matrix[1, 1] = -np.sin(np.deg2rad(par_ang))

    k_vals_image_radec = np.matmul(par_rot_matrix.T, k_vals_image.T).T

    # Convert to L (the Kronecker product of K with it's conjugate).
    # Same math as Jones to Mueller.
    antenna["l_matrix_image_radec"] = jones_to_mueller(k_vals_image_radec)

    # we also need L inverse
    antenna["l_inv_image_radec"] = np.linalg.inv(antenna["l_matrix_image_radec"].T).T

    return antenna, psf

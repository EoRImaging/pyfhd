from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.wcs import WCS
from astropy.time import Time
from astropy import units as u
import numpy as np
from pyuvdata.utils.types import FloatArray


def pressure_from_height(height: float) -> u.Quantity:
    """
    Approximate the pressure from the observatory height.

    This is translated from the IDL astrolib co_refract function and used both
    with the translation of that function below and also when using astropy to
    do the refraction calculation because the astropy altaz frame requires
    pressure & temperature to do the refraction calculation.

    Parameters
    ----------
    height : float
        The height of the observatory in metres above sea level.

    Returns
    -------
    pressure : Quantity
        The estimated pressure based on height.

    """
    pressure = (1010.0 * (1 - 6.5 / 288000.0 * height) ** 5.255) * u.hPa

    return pressure


def temperature_from_height(height: float) -> u.Quantity:
    """
    Approximate the temperature from the observatory height.

    This is translated from the IDL astrolib co_refract function and used both
    with the translation of that function below and also when using astropy to
    do the refraction calculation because the astropy altaz frame requires
    pressure & temperature to do the refraction calculation.

    Parameters
    ----------
    height : float
        The height of the observatory in metres above sea level.

    Returns
    -------
    temperature : Quantity
        The estimated temperature based on height.

    """
    alpha = 0.0065  # temp lapse rate [deg C per meter]
    if height > 11000:
        temperature = 211.5 * u.deg_C
    else:
        temperature = (283.0 - alpha * height) * u.deg_C

    return temperature


def co_refract_forward(
    *,
    alt_in: float | FloatArray,
    pressure: u.Quantity,
    temperature: u.Quantity,
) -> float | FloatArray:
    """
    Calculate refraction correction from observed to true altitude.

    This is a helper function for co_refract.

    Parameters
    ----------
    alt_in : float or array-like of float
        Source altitudes in degrees.
    pressure : Quantity, optional
        Pressure at observatory during observation.
    temperature : Quantity, optional
        Temperature at observatory during observation.

    Returns
    -------
    refraction_correction
        The correction to the refraction to convert observed to true altitudes.

    """
    wh_low = np.nonzero(alt_in < 15.0)[0]
    refraction = 0.0166667 / np.tan(np.deg2rad(alt_in + 7.31 / (alt_in + 4.4)))
    if wh_low.size > 0:
        refraction[wh_low] = (
            3.569
            * (0.1594 + 0.0196 * alt_in[wh_low] + 0.00002 * alt_in[wh_low] ** 2)
            / (1.0 + 0.505 * alt_in[wh_low] + 0.0845 * alt_in[wh_low] ** 2)
        )
    tpcor = pressure.to_value("hPa") / 1010.0 * 283 / temperature.to_value("deg_C")
    refraction_correction = tpcor * refraction

    return refraction_correction


def co_refract(
    alt_in: float | FloatArray,
    height: float,
    pressure: u.Quantity | None = None,
    temperature: u.Quantity | None = None,
    to_observed: bool = False,
    epsilon: float = 0.25,
) -> float | FloatArray:
    """
    Calculate refraction-corrected altitudes.

    Translated from the IDL Astrolib.

    Parameters
    ----------
    alt_in : float or array-like of float
        Source altitudes in degrees.
    height : float
        The height of the observatory in metres above sea level.
    pressure : Quantity, optional
        Pressure at observatory during observation. If None, pressure is
        approximated using the observatory height.
    temperature : Quantity, optional
        Temperature at observatory during observation. If None, temperature is
        approximated using the observatory height.
    to_observed : bool
        Option to invert the calculation to go from observed altitudes to true
        altitudes. This calculation is done iteratively.
    epsilon : float
        Accuracy in arcseconds at which to stop iterating if to_observed is True.

    Returns
    -------
    alt_out : float or array-like of float
        Updated altitudes based on refraction.

    """
    if pressure is None:
        pressure = pressure_from_height(height)
    if temperature is None:
        temperature = temperature_from_height(height=height)

    if not to_observed:
        alt_out = alt_in - co_refract_forward(
            alt_in=alt_in, pressure=pressure, temperature=temperature
        )
    else:
        # calculate initial refraction guess
        dr = co_refract_forward(
            alt_in=alt_in, pressure=pressure, temperature=temperature
        )
        cur = alt_in + dr
        conv_mask = np.full(alt_in.shape, False)
        while True:
            last = cur
            # only update non-converged values
            dr = co_refract_forward(
                alt_in=cur[~conv_mask], pressure=pressure, temperature=temperature
            )
            nonfinite = np.nonzero(~np.isfinite(dr))[0]
            if nonfinite.size > 0:
                dr[nonfinite] = 0
            cur[~conv_mask] = alt_in[~conv_mask] + dr
            conv_mask = np.abs(last - cur) * 3600.0 < epsilon
            if np.all(conv_mask):
                break
        alt_out = cur

    return alt_out


def altaz_to_radec(
    *,
    alt: float,
    az: float,
    lat: float,
    lon: float,
    height: float,
    time: float,
    time_format: str = "jd",
    frame: str = "fk5",
    refraction: bool = False,
    pressure: u.Quantity | None = None,
    temperature: u.Quantity | None = None,
) -> tuple[float, float]:
    """
    Turn Alt/Az coordinates into the equatorial/celestial coordinates RA and DEC.
    The exact location and time must given in order for the coordinates to be calculated.

    Parameters
    ----------
    alt : float
        The altitude in degrees
    az : float
        The azimuth in degrees
    lat : float
        The observatory latitude in degrees.
    lon : float
        The observatory longtitude in degrees.
    height : float
        The height of the observatory in metres above sea level.
    time : float
        The time from the UVFITS file
    time_format : str
        The time format given, as per AstroPy's Time Object, by default jd (Julian)
    frame : str
        The frame for RA and Dec. Must be a frame known to astropy. Defaults to FK5.
    refraction : bool
        Option to account for refraction in earth's atmosphere.
    pressure : Quantity, optional
        Pressure at observatory during observation, only used if refraction is
        True. If None, pressure is approximated using the observatory height.
    temperature : Quantity, optional
        Temperature at observatory during observation, only used if refraction is
        True. If None, temperature is approximated using the observatory height.

    Returns
    -------
    ra : float
        Right Ascension from the given location and time with altitude and azimuth
    dec : float
        Declination from the given location and time with altitude and azimuth
    """

    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height=height * u.meter)
    if refraction:
        if pressure is None:
            pressure = pressure_from_height(height)
        if temperature is None:
            temperature = temperature_from_height(height=height)
    else:
        pressure = 0
        temperature = 0

    altaz = SkyCoord(
        alt=alt * u.deg,
        az=az * u.deg,
        location=loc,
        obstime=Time(time, format=time_format),
        pressure=pressure,
        temperature=temperature,
        frame="altaz",
    )
    radec = altaz.transform_to(frame)

    return radec.ra.deg, radec.dec.deg


def radec_to_altaz(
    ra: float,
    dec: float,
    lat: float,
    lon: float,
    height: float,
    time: float,
    time_format="jd",
    frame: str = "fk5",
    refraction: bool = False,
    pressure: u.Quantity | None = None,
    temperature: u.Quantity | None = None,
) -> tuple[float, float]:
    """
    Turn Celestial/Equatorial coordinates into Alt/Az at the given location and time.
    Time Format by default is Julian, but you can use any of the formats
    provided by the AstroPy Time classes.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    lat : float
        The observatory latitude in degrees.
    lon : float
        The observatory longtitude in degrees.
    height : float
        The height of the observatory in metres above sea level.
    time : float
        The time
    time_format : str, optional
        The format of the time, by default 'jd' (Julian)
    frame : str
        The frame for RA and Dec. Must be a frame known to astropy. Defaults to FK5.
    refraction : bool
        Option to account for refraction in earth's atmosphere.
    pressure : Quantity, optional
        Pressure at observatory during observation, only used if refraction is
        True. If None, pressure is approximated using the observatory height.
    temperature : Quantity, optional
        Temperature at observatory during observation, only used if refraction is
        True. If None, temperature is approximated using the observatory height.

    Returns
    -------
    alt : float
        Altitude coordinates for the given location, time and celestial coordinates
    az : float
        Azimuth coordinates for the given location, time and celestial coordinates
    """

    # Create the Earth Location
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height=height * u.meter)
    loc_time = Time(time, format=time_format)
    radec = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=frame)
    if refraction:
        if pressure is None:
            pressure = pressure_from_height(height)
        if temperature is None:
            temperature = temperature_from_height(height=height)
    else:
        pressure = 0
        temperature = 0

    altaz = radec.transform_to(
        AltAz(
            location=loc, obstime=loc_time, pressure=pressure, temperature=temperature
        )
    )
    return altaz.alt.deg, altaz.az.deg


def radec_to_pixel(
    ra: float,
    dec: float,
    astr: dict,
    *,
    refraction: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    height: float | None = None,
    time: float | None = None,
    time_format="jd",
    pressure: u.Quantity | None = None,
    temperature: u.Quantity | None = None,
) -> tuple[float, float]:
    """
    Turn Celestial Coordinates into Pixel coordinates (X & Y). The astr dictionary should contain
    cdelt, ctype, crpix and crval as per the WCS standard when naxis is 2.

    Parameters
    ----------
    ra : float
        The Right Ascension in degrees.
    dec : float
        The Declination in degrees.
    astr : dict
        The astrometry dictionary used in pyfhd.
    refraction : str, optional
        Choice for how to account for refraction in earth's atmosphere. Can be
        None for no refraction calculation (the default) or "idl" to use IDL
        Astrolib based refraction code or "astropy" to use astropy refraction
        calculation.
    lat : float
        The observatory latitude in degrees. Required if refraction is not None.
    lon : float
        The observatory longtitude in degrees. Required if refraction is not None.
    height : float
        The height of the observatory in metres above sea level. Required if
        refraction is not None.
    time : float
        The time of the observation. Required if refraction is not None.
    time_format : str, optional
        The format of the time, by default 'jd' (Julian). Only used if refraction
        is not None.
    pressure : Quantity, optional
        Pressure at observatory during observation, only used if refraction is
        not None. If None, pressure is approximated using the observatory height.
    temperature : Quantity, optional
        Temperature at observatory during observation, only used if refraction is
        not None. If None, temperature is approximated using the observatory height.

    Returns
    -------
    x : float
        The pixel x coordinate for the given celestial coordinate.
    y : float
        The pixel y coordinate for the given celestial coordinate.
    """
    allowed_refraction = [None, "idl", "astropy"]
    if refraction not in allowed_refraction:
        raise ValueError(
            f"unknown refraction choice. Options are: {allowed_refraction}"
        )

    # Create WCS object with astr
    wcs_astr = WCS(naxis=2)
    wcs_astr.wcs.cdelt = astr["cdelt"]
    # AstroPy ctype requires a list of python string objects, NumPy string objects will not work
    wcs_astr.wcs.ctype = [str(projection) for projection in astr["ctype"]]
    wcs_astr.wcs.crpix = astr["crpix"]
    wcs_astr.wcs.crval = astr["crval"]
    wcs_astr.wcs.equinox = astr["equinox"]
    wcs_astr.wcs.mjdobs = astr["mjdobs"]
    wcs_astr.wcs.dateobs = astr["dateobs"]
    wcs_astr.wcs.radesys = astr["radecsys"]
    wcs_astr.wcs.set_pv(
        [
            (1, 1, astr["pv1"][1]),
            (1, 2, astr["pv1"][2]),
            (1, 3, astr["pv1"][3]),
            (1, 4, astr["pv1"][4]),
            (2, 1, astr["pv2"][0]),
            (2, 2, astr["pv2"][1]),
        ]
    )
    if refraction is not None:
        # convert to Alt/Az with refraction then convert back to RA/Dec without
        # refraction to get the apparent RA/Dec values for the conversion to
        # pixel locations
        if lat is None or lon is None or height is None or time is None:
            raise ValueError(
                "Refraction correction requires lat, lon, height and time to be passed."
            )

        if refraction == "idl":
            alt_deg, az_deg = radec_to_altaz(
                ra=ra,
                dec=dec,
                lat=lat,
                lon=lon,
                height=height,
                time=time,
                time_format=time_format,
                frame=astr["radecsys"].lower(),
                refraction=False,
            )
            alt_deg = co_refract(alt_deg, height=height, to_observed=True)
        else:
            alt_deg, az_deg = radec_to_altaz(
                ra=ra,
                dec=dec,
                lat=lat,
                lon=lon,
                height=height,
                time=time,
                time_format=time_format,
                frame=astr["radecsys"].lower(),
                refraction=True,
                pressure=pressure,
                temperature=temperature,
            )

        ra, dec = altaz_to_radec(
            alt=alt_deg,
            az=az_deg,
            lat=lat,
            lon=lon,
            height=height,
            time=time,
            time_format=time_format,
            frame=astr["radecsys"].lower(),
            refraction=False,
        )

    # Now use world_to_pixel function for WCS objects
    x, y = wcs_astr.world_to_pixel(
        SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=astr["radecsys"].lower())
    )
    # AstroPy returns values as an array, cast to float
    if np.size(x) == 1:
        return float(x), float(y)
    else:
        return x, y


def pixel_to_radec(
    x: float | FloatArray,
    y: float | FloatArray,
    astr: dict,
    *,
    refraction: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    height: float | None = None,
    time: float | None = None,
    time_format="jd",
    pressure: u.Quantity | None = None,
    temperature: u.Quantity | None = None,
) -> tuple[float | FloatArray, float | FloatArray]:
    """
    Turn Pixel coordinates (X & Y) into Celestial Coordinates based off a WCS.
    The astr dictionary should contain cdelt, ctype, crpix and crval as per the
    WCS standard when naxis is 2.

    Parameters
    ----------
    x : float | np.ndarray
        x coordinate(s)
    y : float | np.ndarray
        y coordinate(s)
    astr : dict
        The astrometry dictionary from an obs dictionary
    refraction : str, optional
        Choice for how to account for refraction in earth's atmosphere. Can be
        None for no refraction calculation (the default) or "idl" to use IDL
        Astrolib based refraction code or "astropy" to use astropy refraction
        calculation.
    lat : float
        The observatory latitude in degrees. Required if refraction is not None.
    lon : float
        The observatory longtitude in degrees. Required if refraction is not None.
    height : float
        The height of the observatory in metres above sea level. Required if
        refraction is not None.
    time : float
        The time of the observation. Required if refraction is not None.
    time_format : str, optional
        The format of the time, by default 'jd' (Julian). Only used if refraction
        is not None.
    pressure : Quantity, optional
        Pressure at observatory during observation, only used if refraction is
        not None. If None, pressure is approximated using the observatory height.
    temperature : Quantity, optional
        Temperature at observatory during observation, only used if refraction is
        not None. If None, temperature is approximated using the observatory height.

    Returns
    -------
    ra : float | FloatArray
        Right Ascension from the given pixel coordinates and WCS
    dec : float | FloatArray
        Declination from the given pixel coordinates and WCS

    """
    allowed_refraction = [None, "idl", "astropy"]
    if refraction not in allowed_refraction:
        raise ValueError(
            f"unknown refraction choice. Options are: {allowed_refraction}"
        )

    wcs_astr = WCS(naxis=2)
    wcs_astr.wcs.cdelt = astr["cdelt"]
    # AstroPy ctype requires a list of python string objects, NumPy string objects will not work
    wcs_astr.wcs.ctype = [str(projection) for projection in astr["ctype"]]
    wcs_astr.wcs.crpix = astr["crpix"]
    wcs_astr.wcs.crval = astr["crval"]
    wcs_astr.wcs.equinox = astr["equinox"]
    wcs_astr.wcs.mjdobs = astr["mjdobs"]
    wcs_astr.wcs.dateobs = astr["dateobs"]
    wcs_astr.wcs.radesys = astr["radecsys"]
    wcs_astr.wcs.set_pv(
        [
            (1, 1, astr["pv1"][1]),
            (1, 2, astr["pv1"][2]),
            (1, 3, astr["pv1"][3]),
            (1, 4, astr["pv1"][4]),
            (2, 1, astr["pv2"][0]),
            (2, 2, astr["pv2"][1]),
        ]
    )
    radec = wcs_astr.pixel_to_world(x, y)
    ra = radec.ra
    dec = radec.dec

    if refraction is not None:
        # we have apparent ra/decs. Convert those to apparent alt/az without
        # with no refraction (because that's already in the ra/decs we have)
        # Then convert the apparent alt/az to true ra/dec by applying refraction.
        if lat is None or lon is None or height is None or time is None:
            raise ValueError(
                "Refraction correction requires lat, lon, height and time to be passed."
            )
        alt_deg, az_deg = radec_to_altaz(
            ra=ra,
            dec=dec,
            lat=lat,
            lon=lon,
            time=time,
            time_format=time_format,
            frame=astr["radecsys"].lower(),
            refraction=False,
        )
        if refraction == "idl":
            alt_deg = co_refract(alt_deg, height=height, to_observed=False)

            ra, dec = altaz_to_radec(
                alt=alt_deg,
                az=az_deg,
                lat=lat,
                lon=lon,
                height=height,
                time=time,
                time_format=time_format,
                frame=astr["radecsys"].lower(),
                refraction=False,
            )
        else:
            ra, dec = altaz_to_radec(
                alt=alt_deg,
                az=az_deg,
                lat=lat,
                lon=lon,
                time=time,
                time_format=time_format,
                frame=astr["radecsys"].lower(),
                refraction=True,
                pressure=pressure,
                temperature=temperature,
            )

    return ra, dec

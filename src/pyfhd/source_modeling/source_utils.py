import logging
import time
from datetime import timedelta
from pathlib import Path

from astropy import units
import numpy as np
from pyradiosky import SkyModel

from ..beam_setup.beam_utils import beam_image
from ..pyfhd_tools.pyfhd_utils import (
    angle_difference,
    region_grow,
    resistant_mean,
    spectral_window,
    weight_invert,
)
from ..pyfhd_tools.unit_conv import pixel_to_radec, radec_to_pixel
from ..pyfhd_tools.types import BoolArray, ComplexArray, FloatArray, IntArray

logger = logging.getLogger(__name__)


def generate_source_cal_skymodel(
    *,
    obs: dict,
    psf: dict,
    skymodel: SkyModel | None = None,
    catalog_path: Path | str | None = None,
    sidelobe_catalog_path: Path | str | None = None,
    beam: FloatArray | ComplexArray | None = None,
    mask: BoolArray | None = None,
    allow_sidelobe_sources: bool = False,
    beam_threshold: float | None = None,
    restrict_sources: bool = False,
    flux_threshold: float | None = None,
    no_extend: bool = False,
    max_sources: int | None = None,
    spectral_index: float | None = None,
    preserve_zero_spectral_indices: bool = False,
    flatten_spectrum: bool = False,
    refraction: str | None = None,
) -> SkyModel:
    """
    Make the SkyModel object with the sources needed for visibility modeling.

    Equivalent to the FHD "generate_source_cal_list" function.

    Parameters
    ----------
    obs : dict
        The Observation Metadata dictionary
    psf: dict | h5py.File
        Beam dictionary
    skymodel : SkyModel, optional
        SkyModel object to start with, if provided, catalog_path is ignored.
        Called "source_array" in FHD.
    catalog_path : Path or str, optional
        Path to the catalog to use.
    sidelobe_catalog_path : Path or str, optional
        Path to the catalog to use for the sidelobe sources if different than
        catalog to use for the main beam sources.
    beam : ndarray of float or complex, optional
        Average image-space beam per polarization. Created using `beam_image` if
        not provided. Default is None meaning beam_image will be called.
    mask : ndarray of bool, optional
        Beam mask giving areas to include sources for (True where sources can be
        included, False where sources will be excluded). Default is None.
    allow_sidelobe_sources : bool
        Option to include sidelobe sources. Also affects the defaulting of
        beam_threshold. Default is False.
    beam_threshold : float
        Threshold for beam cut on sources. Sources below the beam_threshold will
        be cut from the skymodel to avoid sources in the nulls. Defaults to 0.05
        unless allow_sidelobe_sources is True, in which case the default is 0.01.
    restrict_sources : bool
        Option to restrict sources to near the beam center. Default is False.
        Related to "no_restrict_model_sources" and "no_restrict_cal_sources" but
        with the opposite sense to avoid double negatives.
    flux_threshold : float, optional
        Threshold for flux values to include. These are catalog fluxes, not
        apparent (i.e. beam-weighted) fluxes. Can be negative, indicating an
        upper bound on fluxes. Default is None.
    no_extend : bool
        Option to replace extended source components with a single component at
        the flux weighted average location with a flux equal to the total flux of
        all the components. Default is False.
    max_sources : int, optional
        Maximum number of sources to include, chosen from highest to lowest
        apparent (i.e. beam-weighted) flux. If a sidelobe_catalog_path is provided,
        sources are taken first from the main lobe catalog and then from the
        sidelobe catalog (if max_sources is greater than the number of sources
        in the main lobe catalog after the various cuts). Default is None.
    spectral_index : float, optional
        Spectral index to use for all sources. Overwrites the spectral index
        from the input catalog. Default is None.
    preserve_zero_spectral_indices : bool
        Option to keep any spectral indices that are set to zero. Default is False,
        If False, the spectral index is reset to the mean spectral index of the
        catalog for any sources with zero spectral index.
    flatten_spectrum : bool
        Option to flatten the spectrum by the average spectral index (calculated
        as a flux-weighted average). Default is False
    refraction : bool
        Option for what refraction algorithm to use to account for refraction in
        earth's atmosphere when computing the pixel locations (and therefore
        when calculating beam values). Allowed values are None (for no refraction
        correction), "idl" to use the refraction algorithm from the IDL astrolib
        or "astropy" to use astropy's refraction algorithm with temperatures and
        pressures estimated using the IDL astrolib algorithm. Default is None.

    Returns
    -------
    skymodel : SkyModel
        SkyModel object containing the sources to be used in visibility modeling.

    """

    if skymodel is None:
        catalog_path = Path(catalog_path)
        if not catalog_path.exists():
            raise ValueError(f"catalog_path does not exist: {catalog_path}")
        skymodel = SkyModel.from_file(catalog_path)

    elif not isinstance(skymodel, SkyModel):
        raise TypeError("If provided, skymodel must be a pyradiosky.SkyModel object.")

    if skymodel.spectral_type != "spectral_index":
        raise NotImplementedError(
            "pyfhd currently only supports spectral index catalogs."
        )

    if skymodel.frame.lower() != "fk5":
        skymodel.transform_to("fk5")

    dimension = obs["dimension"]

    fov = (180 / np.pi) / obs["kpix"]

    freq_use = obs["freq_center"]
    n_pol = obs["n_pol"]

    # use at most xx & yy beams
    n_pol_use = np.min([n_pol, 2])
    if beam is None:
        beam = np.zeros((n_pol_use, dimension, dimension))

        for pol_i in range(n_pol_use):
            beam[pol_i] = beam_image(psf=psf, obs=obs, pol_i=pol_i)

        beam[np.nonzero(beam < 0)] = 0
    else:
        if not isinstance(beam, np.ndarray):
            raise TypeError("beam must be an array.")
        if (
            len(beam.shape) != 3
            or (beam.shape)[0] < np.min([n_pol, 2])
            or (beam.shape)[0] > np.max([n_pol, 2])
            or (beam.shape)[1:] != (dimension, dimension)
        ):
            raise ValueError("beam does not have expected shape.")

    if sidelobe_catalog_path is not None:
        beam_in = beam.copy()
    beam = np.sqrt(np.sum(beam[:n_pol_use] ** 2.0, axis=0) / n_pol_use)

    if beam_threshold is None:
        if allow_sidelobe_sources:
            beam_threshold = 0.01
        else:
            beam_threshold = 0.05
    if beam_threshold > np.max(beam) / 2.0:
        logger.warning(
            f"beam_threshold was set to {beam_threshold}, which is greater than "
            "half the maximum beam value. Using half the maximum beam value for "
            f"the threshold. New beam_threshold is: {np.max(beam) / 2.0}"
        )
        beam_threshold = np.max(beam) / 2.0

    if sidelobe_catalog_path is not None:
        # start at the center of the image (should be close to the beam center)
        obs_i = np.ravel_multi_index((obs["obsx"], obs["obsy"]), (dimension, dimension))
        beam_primary_i = region_grow(beam, obs_i, low=beam_threshold, high=beam.max())
        beam_primary_mask = np.full((dimension, dimension), False, dtype=bool)
        beam_primary_mask.flat[beam_primary_i] = True
        beam_sidelobe_mask = ~beam_primary_mask

        sidelobe_skymodel = generate_source_cal_skymodel(
            obs=obs,
            psf=psf,
            catalog_path=sidelobe_catalog_path,
            beam=beam_in,
            mask=beam_sidelobe_mask,
            allow_sidelobe_sources=True,
            beam_threshold=beam_threshold,
            restrict_sources=restrict_sources,
            flux_threshold=flux_threshold,
            no_extend=no_extend,
            spectral_index=spectral_index,
            preserve_zero_spectral_indices=preserve_zero_spectral_indices,
        )
        allow_sidelobe_sources = False

        del beam_in

    if not restrict_sources:
        fft_alias_range = dimension / 32.0
    else:
        fft_alias_range = dimension / 4.0

    # Add some columns to the object to handle selections on extended sources
    # putting them on the object helps with selects
    skymodel.add_extra_columns(
        names=["ra_deg_use", "dec_deg_use", "flux_I_use"],
        values=[skymodel.ra.deg, skymodel.dec.deg, skymodel.stokes[0, 0]],
    )

    # handle extended sources
    # ra/dec cuts should be on flux weighted average of extended component locations
    if skymodel.extended_model_group is not None:
        extended_comps = np.nonzero(skymodel.extended_model_group != "")[0]
        if extended_comps.size == 0:
            skymodel.extended_model_group = None

    if skymodel.extended_model_group is not None:
        extended_srcs = np.unique(skymodel.extended_model_group[extended_comps])
        ext_src_lists = {}
        for src in extended_srcs:
            wh_src = np.nonzero(skymodel.extended_model_group == src)
            ext_src_lists[src] = wh_src

            ra_vals = skymodel.extra_columns["ra_deg_use"][wh_src]
            if np.max(ra_vals) - np.min(ra_vals) > np.pi:
                # there's a branch cut in the middle of this extended source
                ra_vals[ra_vals > np.pi] -= 360

            avg_ra = np.average(
                ra_vals, weights=skymodel.extra_columns["flux_I_use"][wh_src]
            )
            if avg_ra < 0:
                avg_ra += 360
            skymodel.extra_columns["ra_deg_use"][wh_src] = avg_ra
            avg_dec = np.average(
                skymodel.extra_columns["dec_deg_use"][wh_src],
                weights=skymodel.extra_columns["flux_I_use"][wh_src],
            )
            skymodel.extra_columns["dec_deg_use"][wh_src] = avg_dec

            total_I_flux = skymodel.extra_columns["flux_I_use"][wh_src].sum()
            skymodel.extra_columns["flux_I_use"][wh_src] = total_I_flux

    ra0 = obs["obsra"]
    dec0 = obs["obsdec"]
    angs = angle_difference(
        ra1=ra0,
        dec1=dec0,
        ra2=skymodel.extra_columns["ra_deg_use"],
        dec2=skymodel.extra_columns["dec_deg_use"],
        degree=True,
    )
    i_use = np.nonzero(np.abs(angs) < fov / 2.0)[0]
    n_use = i_use.size

    if spectral_index is not None:
        skymodel.spectral_index[:] = spectral_index
    elif not preserve_zero_spectral_indices:
        zero_i = np.nonzero(skymodel.spectral_index == 0)[0]
        n_zero = zero_i.size
        nonzero_i = np.nonzero(skymodel.spectral_index != 0)[0]
        n_nonzero = nonzero_i.size
        if n_zero > 0:
            if n_nonzero > 5:
                alpha_mean = resistant_mean(
                    skymodel.spectral_index[nonzero_i], deviations=2
                )
            else:
                alpha_mean = -0.8
            skymodel.spectral_index[zero_i] = alpha_mean

    if n_use > 0:
        skymodel.select(component_inds=i_use)
        x_arr, y_arr = radec_to_pixel(
            ra=skymodel.ra.deg,
            dec=skymodel.dec.deg,
            astr=obs["astr"],
            refraction=refraction,
            lat=obs["lat"],
            lon=obs["lon"],
            height=obs["alt"],
            time=obs["jd0"],
        )
        skymodel.add_extra_columns(names=["image_x", "image_y"], values=[x_arr, y_arr])

        if allow_sidelobe_sources:
            beam_i = np.nonzero(beam > beam_threshold)
        else:
            image_center = np.ravel_multi_index(
                (dimension // 2, dimension // 2), (dimension, dimension)
            )
            thresh_low = np.min([np.max(beam) / 2.0, beam_threshold])
            beam_i = region_grow(beam, image_center, low=thresh_low, high=beam.max())
            beam_i = np.unravel_index(beam_i, (dimension, dimension))

        beam_mask = np.full((dimension, dimension), False, dtype=bool)
        beam_mask[beam_i] = True
        if mask is not None and mask.shape == beam_mask.shape:
            beam_mask *= mask

        # Add some columns to the object to handle selections on extended sources
        # putting them on the object helps with selects
        # spectral_index type, so Nfreqs=1
        skymodel.add_extra_columns(
            names=["x_use", "y_use"],
            values=[
                skymodel.extra_columns["image_x"],
                skymodel.extra_columns["image_y"],
            ],
        )

        # if no extended sources survive the cuts, set extended_model_group to None
        if skymodel.extended_model_group is not None:
            extended_comps = np.nonzero(skymodel.extended_model_group != "")[0]
            if extended_comps.size == 0:
                skymodel.extended_model_group = None

        # flux cuts should be on sum of extended components
        # x/y cuts should be on flux weighted average of extended component locations
        if skymodel.extended_model_group is not None:
            extended_srcs = np.unique(skymodel.extended_model_group[extended_comps])
            ext_src_lists = {}
            for src in extended_srcs:
                wh_src = np.nonzero(skymodel.extended_model_group == src)
                ext_src_lists[src] = wh_src[0]

                avg_x = np.average(
                    skymodel.extra_columns["x_use"][wh_src],
                    weights=skymodel.extra_columns["flux_I_use"][wh_src],
                )
                skymodel.extra_columns["x_use"][wh_src] = avg_x
                avg_y = np.average(
                    skymodel.extra_columns["y_use"][wh_src],
                    weights=skymodel.extra_columns["flux_I_use"][wh_src],
                )
                skymodel.extra_columns["y_use"][wh_src] = avg_y

                total_I_flux = skymodel.extra_columns["flux_I_use"][wh_src].sum()
                skymodel.extra_columns["flux_I_use"][wh_src] = total_I_flux

            if no_extend:
                keep_comp = np.full((skymodel.Ncomponents,), True)
                # convert extended models into a single component at the flux
                # weighted average location with the total flux
                for comp_arr in ext_src_lists.values():
                    first_comp = comp_arr[0]
                    skymodel.stokes[:, :, first_comp] = skymodel.stokes[
                        :, :, comp_arr
                    ].sum(axis=-1)
                    ra, dec = pixel_to_radec(
                        x=skymodel.extra_columns["x_use"][first_comp],
                        y=skymodel.extra_columns["y_use"][first_comp],
                        astr=obs["astr"],
                    )
                    skymodel.ra[first_comp] = ra
                    skymodel.dec[first_comp] = dec
                    skymodel.extended_model_group[first_comp] = ""

                    keep_comp[comp_arr[1:]] = False

                skymodel.select(component_inds=np.nonzero(keep_comp)[0])
                skymodel.extended_model_group = None

        if flux_threshold is not None:
            flux_I_use = skymodel.extra_columns["flux_I_use"]
            if flux_threshold < 0:
                # interpret negative flux thresholds as upper bounds.
                # Weird, but what FHD does
                flux_I_use *= -1

            src_use = np.nonzero(
                (skymodel.extra_columns["x_use"] >= fft_alias_range)
                & (skymodel.extra_columns["x_use"] <= dimension - 1 - fft_alias_range)
                & (skymodel.extra_columns["y_use"] >= fft_alias_range)
                & (skymodel.extra_columns["y_use"] <= dimension - 1 - fft_alias_range)
                & (flux_I_use > flux_threshold)
                & (skymodel.extra_columns["flux_I_use"] != 0)
            )[0]

        else:
            src_use = np.nonzero(
                (skymodel.extra_columns["x_use"] >= fft_alias_range)
                & (skymodel.extra_columns["x_use"] <= dimension - 1 - fft_alias_range)
                & (skymodel.extra_columns["y_use"] >= fft_alias_range)
                & (skymodel.extra_columns["y_use"] <= dimension - 1 - fft_alias_range)
                & (skymodel.extra_columns["flux_I_use"] != 0)
            )[0]
        n_src_use = src_use.size
        if n_src_use == 0:
            logger.warning("No sources in model catalog image range and flux range.")
            skymodel = None
        else:
            src_use2 = np.nonzero(
                beam_mask[
                    skymodel.extra_columns["x_use"][src_use].round().astype(int),
                    skymodel.extra_columns["y_use"][src_use].round().astype(int),
                ]
            )[0]
            n_src_use = src_use2.size
            if n_src_use > 0:
                src_use = src_use[src_use2]

            skymodel.select(component_inds=src_use)

            inds_finite = np.nonzero(np.isfinite(skymodel.extra_columns["flux_I_use"]))[
                0
            ]
            n_finite = inds_finite.size
            if n_finite == 0:
                logger.warning(
                    "All sources in model catalog after image and flux cuts are "
                    "nan/inf."
                )
                skymodel = None
            else:
                if n_finite != skymodel.Ncomponents:
                    logger.warning(
                        "Model catalog contains nan/inf fluxes, dropping them."
                    )
                    skymodel.select(component_inds=inds_finite)
                    n_src_use = n_finite

                # if no extended sources survive the cuts, set
                # extended_model_group to None
                if skymodel.extended_model_group is not None:
                    extended_comps = np.nonzero(skymodel.extended_model_group != "")[0]
                    if extended_comps.size == 0:
                        skymodel.extended_model_group = None

                # calculate apparent flux given the beam
                # NB: IDL FHD uses float indices into beam, which results in a
                # truncation of the float to an int. Here we use a round, but
                # it should be a very small difference (and only affects sorting
                # unless cutting on number of sources)
                skymodel.add_extra_columns(
                    names=["beam_I"],
                    values=beam[
                        skymodel.extra_columns["x_use"].round().astype(int),
                        skymodel.extra_columns["y_use"].round().astype(int),
                    ],
                )

                influence = (
                    skymodel.extra_columns["flux_I_use"]
                    * skymodel.extra_columns["beam_I"]
                )

                # remove the extra columns just used internally
                skymodel.remove_extra_columns(
                    ["ra_deg_use", "dec_deg_use", "x_use", "y_use", "flux_I_use"]
                )

                # sort from max to min apparent flux
                order = np.flip(np.argsort(influence))
                skymodel._select_along_param_axis({"Ncomponents": order})

    if flatten_spectrum:
        wh_pos_I_flux = np.nonzero(skymodel.stokes[0, 0] > 0)[0]
        alpha_avg = np.average(
            skymodel.spectral_index[wh_pos_I_flux],
            weights=skymodel.stokes[0, 0, wh_pos_I_flux],
        )
        obs["alpha_avg"] = alpha_avg
        skymodel.spectral_index -= alpha_avg

    # call `at_frequencies` method to get it at the central obs freq:
    skymodel.at_frequencies(np.atleast_1d(freq_use) * units.Hz)

    if sidelobe_catalog_path is not None:
        if skymodel is not None:
            skymodel.concat(sidelobe_skymodel)
        else:
            logger.warning(
                "No remaining sources in model catalog after cuts, only using "
                "sidelobe catalog."
            )
            skymodel = sidelobe_skymodel

    if skymodel is None:
        return None

    if max_sources is not None:
        if skymodel.extended_model_group is None:
            comp_keep = np.arange(max_sources)
        else:
            extended_comps = np.nonzero(skymodel.extended_model_group != "")[0]
            if max_sources < extended_comps.min():
                # no extended sources needed
                comp_keep = np.arange(max_sources)
            else:
                extended_mask = np.full((skymodel.Ncomponents,), False)
                extended_mask[extended_comps] = True
                compact_src_inds = np.nonzero(skymodel.extended_model_group == "")[0]

                # get the extended sources in their original order
                extended_srcs, extended_start_inds = np.unique(
                    skymodel.extended_model_group[extended_comps], return_index=True
                )

                ext_src_lists = {}
                for src in extended_srcs:
                    wh_src = np.nonzero(skymodel.extended_model_group == src)[0]
                    ext_src_lists[src] = wh_src

                src_inds_select = np.sort(
                    np.concatenate((compact_src_inds, extended_start_inds))
                )[:max_sources]
                compact_keep = np.intersect1d(compact_src_inds, src_inds_select)
                extended_start_keep = np.intersect1d(
                    extended_start_inds, src_inds_select
                )

                extended_keep = np.concatenate(
                    [
                        ext_src_lists[skymodel.extended_model_group[src]]
                        for src in extended_start_keep
                    ]
                )

                comp_keep = np.sort(np.concatenate((compact_keep, extended_keep)))

        skymodel.select(component_inds=comp_keep)

    return skymodel


def stokes_cnv(
    sky: FloatArray | ComplexArray | SkyModel,
    *,
    antenna: dict,
    obs: dict,
    beam_arr: ComplexArray | None = None,
    inverse: bool = False,
    square: bool = False,
) -> FloatArray | ComplexArray | SkyModel:
    """
    Convert fluxes between Stokes and instrumental pols.

    Accepts either an image array (with a pol axis) or a Skymodel object. Uses
    the feed_aligned_projection on the antenna object to project between RA/Dec
    aligned polarizations (for Stokes) and feed aligned polarizations (for
    instrumental pol). The forward (default) direction goes from instrumental
    pol to Stokes, set `inverse=True` to go from Stokes to instrumental pol.

    NB: the `no_extend` keyword is not implemented here because given they way
    extended sources are handled in SkyModel objects, skipping them doesn't save
    any time (in fact it would slow things down). In FHD, the `no_extend` keyword
    means that the conversion is only done on the top level sources (so effectively
    the point source equivalent of the extended source). It seems like that requires
    that the downstream code also only uses the top level sources (because the
    lower level ones remain in the structure but are not updated).

    Several apparent debugging options (rotate_pol, no_dipole_projection_rotation,
    center_rotate, debugging_direction) which are not used anywhere in the codebase
    were also not implemented here.

    Parameters
    ----------
    sky : np.ndarray or SkyModel
        Either an array containing images (shape: (n_pol, image dimension,
        image dimension)) or a SkyModel object. If a SkyModel and inverse is
        False, must have extra columns giving instrumental pol fluxes.
    antenna : dict
        Antenna dict containing the feed_aligned_projection matrix.
    obs : dict
        Observation dict.
    beam_arr : np.ndarray, optional
        Either the image space beam to use to adjust fluxes based on beam sensitivity,
        usually generated using the `beam_image` function (shape: (n_pol,
        image dimension, image dimension)) or, if sky is a SkyModel object, an
        array of pre-calculated beam sensitivities for each component
        (shape: (n_pol, sky.Ncomponents)). If None, no flux correction is applied.
        Default is None.
    inverse : bool
        Option to go from Stokes to instrumental pol. Default is False.
    square : bool
        Option to use the square of the beam_arr for the flux correction.
        Default is False.

    Returns
    -------
    np.ndarray or SkyModel
        Returns the same type as sky. If a SkyModel and inverse is True, the
        instrumental polarization fluxes are added in extra columns.

    """
    n_pol = None
    if not isinstance(sky, SkyModel):
        if not isinstance(sky, np.ndarray) or sky.ndim != 3 or sky.shape[0] > 4:
            raise ValueError(
                "sky can be a pyradiosky SkyModel object or a 3 dimensional "
                "array with the zeroth axis as the polarization axis."
            )
        n_pol = sky.shape[0]

    if beam_arr is None:
        n_pol = obs["n_pol"]
        beam_use = np.ones((n_pol, obs["dimension"], obs["elements"]), dtype=float)
    else:
        if isinstance(sky, SkyModel):
            allowed_beam_ndim = [2, 3]
        else:
            allowed_beam_ndim = [3]
        allowed_ndim_str = " or ".join([str(ndim) for ndim in allowed_beam_ndim])
        if (
            not isinstance(beam_arr, np.ndarray)
            or beam_arr.ndim not in allowed_beam_ndim
            or beam_arr.shape[0] > 4
        ):
            raise ValueError(
                "beam_arr must be an array with the zeroth axis as the "
                f"polarization axis and {allowed_ndim_str} dimensions."
            )

        if n_pol is None:
            n_pol = beam_arr.shape[0]
            beam_use = beam_arr
        elif beam_arr.shape[0] >= n_pol:
            beam_use = beam_arr[:n_pol]
        else:
            raise ValueError("beam_arr has fewer polarizations than input sky.")

        if square:
            beam_use = beam_use**2

    n_pix = antenna["image_pix_use"].size

    # Use L when going from Stokes to instrument (inverse=True)
    # use L inverse when going from instrument to Stokes
    # Note: this is opposite of FHD's use of L and L inverse but I believe that
    # it is correct and the FHD implementation is wrong.
    if inverse:
        p_use = antenna["l_matrix_image_radec"]
    else:
        p_use = antenna["l_inv_image_radec"]

    # Define the Stokes conversion:
    # I = xx* + yy*
    # Q = xx* - yy*
    # U = xy* + yx*
    # V = ixy* - iyx*
    # where x is in the RA direction and y is in the Dec direction
    stokes_mat = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j, -0.0 - 1.0j],
        ]
    )

    stokes_inv = np.linalg.inv(stokes_mat)

    if isinstance(sky, SkyModel):
        if beam_use.ndim == 3:
            sx = sky.extra_columns["image_x"]
            sy = sky.extra_columns["image_y"]
            # NB: FHD just uses sx, sy as indices, which means they are truncated
            # to ints. We will use round
            sx = sx.round().astype(int)
            sy = sy.round().astype(int)

            # set background to -1 to catch out of range pixels
            ind_arr = np.zeros((obs["dimension"], obs["elements"]), dtype=int) - 1
            ind_arr.flat[antenna["image_pix_use"]] = np.arange(n_pix)
            # NB: FHD just uses sx, sy as indices, which means they are truncated
            # to ints. We will use round
            p_ind = ind_arr[sx, sy]
            s_use = np.nonzero(p_ind > 0)[0]
            if s_use.size == 0:
                raise ValueError("Error: probably no sources above the horizon")
            sx = sx[s_use]
            sy = sy[s_use]
            p_ind = p_ind[s_use]
            beam_use = beam_use[:, sx, sy]

        if inverse:
            # Stokes -> instrumental
            flux_radec_coherency = np.matmul(stokes_inv, sky.stokes[:, 0, s_use].value)

            # Need a matrix multiply per source. I can't figure out how to do it
            # with just numpy matrix multiplies right now because everything
            # I tried ended up with 2 source axes (since both p_use and
            # fluxes have source axes).
            # We could iterate over the source axis but that's slow if there are
            # a lot of sources. Iterate over the radec & pol axes instead.
            flux_feed_aligned = np.zeros((n_pol, s_use.size), dtype=np.complex128)
            for rd_i in range(4):
                for pol_i in range(n_pol):
                    flux_feed_aligned[pol_i, :] += (
                        p_use[rd_i, pol_i, p_ind] * flux_radec_coherency[rd_i]
                    )

            flux_feed_aligned *= beam_use
            extra_cols = {}
            for pol_i in range(n_pol):
                extra_cols[f"flux_pol_{pol_i}"] = flux_feed_aligned[pol_i]
            sky.add_extra_columns(names=extra_cols.keys(), values=extra_cols.values())
        else:
            # instrumental -> Stokes
            # assume the instrumental fluxes are in extra_columns
            col_names = [f"flux_pol_{pol_i}" for pol_i in range(n_pol)]

            # Need a matrix multiply per source. I can't figure out how to do it
            # with just numpy matrix multiplies right now because everything
            # I tried ended up with 2 source axes (since both p_use and
            # fluxes have source axes).
            # We could iterate over the source axis but that's slow if there are
            # a lot of sources. Iterate over the radec & pol axes instead.
            flux_radec_coherency = np.zeros((4, s_use.size), dtype=np.complex128)
            for rd_i in range(4):
                for pol_i in range(n_pol):
                    flux_radec_coherency[pol_i, :] += (
                        p_use[pol_i, rd_i, p_ind]
                        * weight_invert(beam_use[pol_i])
                        * sky.extra_columns[col_names[pol_i]]
                    )

            flux_stokes = np.matmul(stokes_mat, flux_radec_coherency)

            # set polarizations not supported by data to zero
            if n_pol < 4:
                flux_stokes[:n_pol] = 0

            sky.stokes[:, 0, :] = flux_stokes * units.Jy
        return sky

    # sky is an image array
    # redefine n_pol here, just to make sure it matches the images
    n_pol = sky.shape[0]

    if inverse:
        # Stokes -> instrumental
        image_arr_radec_coherency = np.matmul(stokes_inv, sky)

        # Need a matrix multiply per source. I can't figure out how to do it
        # with just numpy matrix multiplies right now because everything
        # I tried ended up with 2 source axes (since both p_use and
        # fluxes have source axes).
        # We could iterate over the source axis but that's slow if there are
        # a lot of sources. Iterate over the radec & pol axes instead.
        image_arr_feed_aligned = np.zeros(
            (n_pol, sky.shape[1], sky.shape[2]), dtype=np.complex128
        )
        for rd_i in range(4):
            for pol_i in range(n_pol):
                image_arr_feed_aligned[pol_i] += (
                    p_use[rd_i, pol_i] * image_arr_radec_coherency[rd_i]
                )
        image_arr_feed_aligned *= beam_use
    else:
        # instrumental -> Stokes

        # make this have 4 polarizations no matter how many are actually
        # present so that matrix multiplies work properly.
        # zero polarizations not supported by data after the matrix multiplies
        image_arr_feed_aligned = np.zeros(
            (4, sky.shape[1], sky.shape[2]), dtype=np.complex128
        )
        image_arr_feed_aligned[:n_pol] = weight_invert(beam_use) * sky

        # Need a matrix multiply per source. I can't figure out how to do it
        # with just numpy matrix multiplies right now because everything
        # I tried ended up with 2 source axes (since both p_use and
        # fluxes have source axes).
        # We could iterate over the source axis but that's slow if there are
        # a lot of sources. Iterate over the radec & pol axes instead.
        image_arr_radec_coherency = np.zeros(
            (4, sky.shape[1], sky.shape[2]), dtype=np.complex128
        )
        for rd_i in range(4):
            for pol_i in range(n_pol):
                image_arr_radec_coherency[pol_i, :] += (
                    p_use[pol_i, rd_i, p_ind] * image_arr_feed_aligned[pol_i]
                )

        image_arr_stokes = np.matmul(stokes_mat, flux_radec_coherency)

        # drop polarizations not supported by data
        if n_pol < 4:
            image_arr_stokes = image_arr_stokes[:n_pol]

        return image_arr_stokes


def _setup_uv_vals(
    *,
    dimension: int,
    elements: int,
    xvals: FloatArray | None = None,
    yvals: FloatArray | None = None,
    uv_i_use: tuple[IntArray] | None = None,
):
    """Set up uv locations to DFT to."""
    if uv_i_use is not None:
        if xvals is not None or yvals is not None:
            raise ValueError("either pass uv_i_use or xvals and yvals not both.")
        xvals = uv_i_use[0] - dimension / 2.0
        yvals = uv_i_use[1] - elements / 2.0
    elif xvals is not None or yvals is not None:
        if xvals is None or yvals is None:
            raise ValueError("If xvals or yvals is provided they must both be provided")
        if xvals.shape != yvals.shape:
            raise ValueError("xvals and yvals must have the same shape")
    else:
        xvals, yvals = np.meshgrid(
            np.arange(dimension) - dimension / 2.0,
            np.arange(elements) - elements / 2.0,
            indexing="ij",
        )
        xvals = xvals.flatten()
        yvals = yvals.flatten()
    return xvals, yvals


def source_dft(
    *,
    x_loc: FloatArray,
    y_loc: FloatArray,
    dimension: int,
    elements: int,
    flux: FloatArray | ComplexArray,
    xvals: FloatArray | None = None,
    yvals: FloatArray | None = None,
    inds_use: IntArray | None = None,
    conserve_memory: bool = True,
    memory_threshold: float = 1e8,
) -> ComplexArray:
    """
    DFT sources to a model uv plane.

    Gaussian source models not yet supported.

    Parameters
    ----------
    x_loc : np.ndarray of float
        Source x locations (in pixel units).
    y_loc : np.ndarray of float
        Source y locations (in pixel units).
    dimension : int
        Size of image (x dimension).
    elements : int
        Size of image (y dimension).
    flux : np.ndarray of float
        Fluxes for sources, shape: (n_pol, n_sources)
    xvals : np.ndarray of float
        u pixel values in uv plane to DFT to.
    yvals : np.ndarray of float
        v pixel values in uv plane to DFT to.
    inds_use : np.ndarray of int, optional
        Indices to use from source arrays.
    conserve_memory : bool
        Option to limit memory use by chunking the number of sources to DFT
        at a time.
    memory_threshold : float
        Memory threshold, which sets the number of sources to DFT at once if
        conserve_memory is True. Default is 1e8.

    Returns
    -------
    model_uv_arr : ndarray of complex
        Model uv plane.

    """
    xvals, yvals = _setup_uv_vals(
        dimension=dimension, elements=elements, xvals=xvals, yvals=yvals
    )
    n_pol = flux.shape[0]

    if inds_use is not None:
        x_loc_use = x_loc[inds_use]
        y_loc_use = y_loc[inds_use]
        flux_use = flux[inds_use]
    else:
        x_loc_use = x_loc
        y_loc_use = y_loc
        flux_use = flux

    x_use = x_loc_use - dimension / 2.0
    y_use = y_loc_use - elements / 2.0
    x_use *= 2.0 * np.pi / dimension
    y_use *= 2.0 * np.pi / elements

    n_src = x_use.size
    if n_src == 1:
        conserve_memory = False

    n_calc = xvals.size * n_src
    mem_factor = 32.0  # from rough empirical estimations
    if conserve_memory and (n_calc * mem_factor > memory_threshold):
        # DFT with memory management
        # If the max memory is less than the estimated memory needed to DFT all
        # sources at once, then break the DFT into chunks
        sources_per_bin = int(
            np.round(n_src / np.ceil(n_calc * mem_factor / memory_threshold))
        )
        sources_per_bin = max([sources_per_bin, 1])
        memory_bins = int(np.ceil(n_src / sources_per_bin))

        binsize = np.full(memory_bins, sources_per_bin, dtype=int)
        # Last bin may contain less sources than the other bins (number of bins
        # may not evenly divide the number of sources)
        binsize[-1] -= np.sum(binsize) - n_src

        # Index of the first source in each bin
        bin_start = np.cumsum(binsize) - binsize
    else:
        memory_bins = 1
        binsize = [n_src]
        bin_start = [0]

    source_uv_vals = np.zeros((n_pol, xvals.size), dtype=np.complex128)
    t0 = time.time()
    reporting_frac = 0.2
    logger.info(
        f"DFT setup complete, begin DFT calculation ({n_src} sources in "
        f"{memory_bins} steps)"
    )
    for bin_i in range(memory_bins):
        inds = np.arange(binsize[bin_i]) + bin_start[bin_i]
        # Calculate sin and cosine of exponential in DFT (faster than a direct exp)
        phase = np.outer(x_use[inds], xvals) + np.outer(y_use[inds], yvals)
        cos_term = np.cos(phase)
        sin_term = np.sin(phase)
        del phase
        source_uv_vals += np.matmul(flux_use[:, inds], cos_term) + 1j * np.matmul(
            flux_use[:, inds], sin_term
        )
        loop_time = time.time()
        if (
            memory_bins > int(1.0 / reporting_frac)
            and bin_i % int(np.round(memory_bins * reporting_frac)) == 0
            and (bin_i + 1) < memory_bins
        ):
            ave_loop_time = (loop_time - t0) / (bin_i + 1)
            est_time_left = timedelta(
                seconds=round((memory_bins - (bin_i + 1)) * ave_loop_time, 2)
            )
            logger.info(
                f"{bin_i + 1}/{memory_bins} DFT loops completed. Average loop "
                f"time: {round(ave_loop_time, 3)} seconds. Estimated time "
                f"remaining: {est_time_left}"
            )

    del cos_term, sin_term

    return source_uv_vals


def source_dft_multi(
    *,
    obs: dict,
    antenna: dict,
    skymodel: SkyModel,
    uv_i_use: tuple[IntArray] | None = None,
    conserve_memory: bool = True,
    memory_threshold: float = 1e8,
) -> ComplexArray:
    """
    DFT multiple sources to a model uv plane.

    This is the high-level function that in FHD can either do a DFT or an FFT
    approximation (which is much faster.)
    This currently just does a DFT. In future we plan enable using an FFT with
    the DFT approximation as in FHD.

    Gaussian source models not yet supported.

    Parameters
    ----------
    obs : dict
        The observation metadata dictionary.
    antenna : dict
        The antenna/beam dictionary.
    skymodel : pyradiosky.SkyModel
        Skymodel object containing the sources to use.
    uv_i_use : tuple of np.ndarray of int
        Tuple of index arrays giving the locations in the uv plane to use.
    sigma_threshold : float, optional
        Signal to noise threshold on included sources.
    conserve_memory : bool
        Option to limit memory use by chunking the number of sources to DFT
        at a time.
    memory_threshold : float
        Memory threshold, which sets the number of sources to DFT at once if
        conserve_memory is True. Default is 1e8.

    Returns
    -------
    model_uv_arr : ndarray of complex
        Model uv plane.

    """
    n_pol = obs["n_pol"]
    dimension = obs["dimension"]
    elements = obs["elements"]
    n_spectral = obs["degrid_spectral_terms"]
    if n_spectral != 0:
        raise NotImplementedError("degridding spectral terms is not yet implemented.")

    xvals, yvals = _setup_uv_vals(
        dimension=dimension, elements=elements, uv_i_use=uv_i_use
    )

    x_vec = skymodel.extra_columns["image_x"]
    y_vec = skymodel.extra_columns["image_y"]

    sky_use = stokes_cnv(skymodel, antenna=antenna, obs=obs, inverse=True)
    flux_arr = np.zeros((n_pol, sky_use.Ncomponents), dtype=np.complex128)
    for pol_i in range(n_pol):
        flux_arr[pol_i] = sky_use.extra_columns[f"flux_pol_{pol_i}"]

    if (
        sky_use.spectral_type != "full"
        or sky_use.Nfreqs != 1
        or not np.isclose(obs["freq_center"], sky_use.freq_array[0].value)
    ):
        raise ValueError("skymodel is expected to already match central obs frequency")

    logger.info("Creating source model as single continuum uv plane")

    model_uv_full = np.zeros((n_pol, dimension, elements), dtype=np.complex128)
    model_uv_vals = source_dft(
        x_loc=x_vec,
        y_loc=y_vec,
        xvals=xvals,
        yvals=yvals,
        dimension=dimension,
        elements=elements,
        flux=flux_arr,
        memory_threshold=memory_threshold,
        conserve_memory=conserve_memory,
    )

    model_uv_full[:, uv_i_use[0], uv_i_use[1]] = model_uv_vals

    return model_uv_full


def source_dft_model(
    *,
    obs: dict,
    antenna: dict,
    skymodel: SkyModel,
    uv_mask: BoolArray | None = None,
    sigma_threshold: float | None = None,
    conserve_memory: bool = True,
    memory_threshold: float = 1e8,
) -> ComplexArray:
    """
    Coordinate DFTing sources to a model uv plane.

    This is the high-level function that in FHD can either do a DFT or an FFT
    approximation (which is much faster.)
    This currently just does a DFT. In future we plan enable using an FFT with
    the DFT approximation as in FHD.

    Gaussian source models not yet supported.

    Parameters
    ----------
    obs : dict
        The observation metadata dictionary.
    antenna : dict
        The antenna/beam dictionary.
    skymodel : pyradiosky.SkyModel
        Skymodel object containing the sources to use.
    uv_mask : ndarray of bool, optional
        Boolean mask of what parts of the uv plane to use. Defaults to using
        the whole plane.
    sigma_threshold : float, optional
        Signal to noise threshold on included sources.
    conserve_memory : bool
        Option to limit memory use by chunking the number of sources to DFT
        at a time.
    memory_threshold : float
        Memory threshold, which sets the number of sources to DFT at once if
        conserve_memory is True. Default is 1e8.

    Returns
    -------
    model_uv_arr : ndarray of complex
        Model uv plane.

    """
    dimension = obs["dimension"]
    elements = obs["elements"]

    if uv_mask is None:
        uv_mask = np.full((dimension, elements), True, dtype=bool)

    uv_i_use = np.nonzero(uv_mask)

    if sigma_threshold is not None:
        raise NotImplementedError(
            "Signal to Noise thresholding is not yet implemented."
        )

    model_uv_arr = source_dft_multi(
        obs=obs,
        antenna=antenna,
        skymodel=skymodel,
        uv_i_use=uv_i_use,
        conserve_memory=conserve_memory,
        memory_threshold=memory_threshold,
    )

    return model_uv_arr


def vis_delay_filter(
    vis_model_arr: ComplexArray, *, obs: dict, params: dict
) -> ComplexArray:
    """
    Apply a delay space filter at the horizon to remove fft artifacts.

    Model visibilities are phased to zenith, windowed, transformed to delay space,
    cut at the horizon, transformed back to visibility space, unwindowed, unphased
    from zenith, and frequency cut to match the data.

    Parameters
    ----------
    vis_model_arr : ndarray of complex
        Input model visibilities to be filtered.
    obs : dict
        The observation metadata dictionary.
    params : dict
        Visibility metadata dictionary.

    Returns
    -------
    vis_model_arr : ndarray of complex
        Delay filtered visibilities with half the number of frequencies as the
        input visibilities.
    """
    # u,v,w are in light travel time in seconds
    freq_arr = obs["baseline_info"]["freq"]
    freq_res = obs["freq_res"]
    n_pol = obs["n_pol"]
    kbinsize = obs["kpix"]
    nfreq = freq_arr.size
    nbl = params["uu"].size

    data = vis_model_arr.transpose(1, 2, 0)

    # test with removing zeroed visibilities instead
    total_data = np.sum(np.abs(data), axis=(0, 2))
    bi_use = np.nonzero(total_data != 0)
    cross_inds = np.nonzero(params["antenna1"] != params["antenna2"])
    bi_use = np.intersect1d(bi_use, cross_inds)

    data = data[:, bi_use]
    uu = params["uu"][bi_use]
    vv = params["vv"][bi_use]
    ww = params["ww"][bi_use]
    bb = np.sqrt(uu**2.0 + vv**2.0 + ww**2.0)
    nbl = bi_use.size

    # Phase to zenith -- easier calculations of the location of the horizon when
    # phased to zenith
    dimension = obs["dimension"]
    dx = obs["obsx"] - obs["zenx"]
    dy = obs["obsy"] - obs["zeny"]
    dx *= 2.0 * np.pi / dimension
    dy *= 2.0 * np.pi / dimension
    phase = (
        np.outer(freq_arr, uu) * dx / kbinsize + np.outer(freq_arr, vv) * dy / kbinsize
    )
    rephase_vals = np.cos(phase) + 1j * np.sin(phase)
    rephase_vals = np.repeat((rephase_vals[:, :, np.newaxis]), n_pol, axis=2)
    data *= rephase_vals
    del uu, vv, ww

    # Apply window function
    window = spectral_window(nfreq, type="Blackman-Harris", periodic=True)
    norm_factor = np.sqrt(nfreq / np.sum(window**2.0))
    window = window * norm_factor
    window_expand = np.repeat(
        np.repeat(window[:, np.newaxis, np.newaxis], nbl, axis=1), n_pol, axis=2
    )
    data = data * window_expand

    # FFT along freq axis
    spectra = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    del data, window

    # Cut at the horizon
    tau_cut = 1.0

    # Calculate upper and lower delay limits for each baseline
    lower_limit = nfreq * (0.5 - bb * tau_cut * freq_res)
    upper_limit = nfreq * (0.5 + bb * tau_cut * freq_res)

    for freq_i in range(nfreq):
        mask_high = freq_i / upper_limit
        mask_high_inds = np.nonzero(mask_high > 1.0)
        if mask_high_inds[0].size > 0:
            spectra[freq_i, mask_high_inds] = 0

        mask_low = freq_i / lower_limit
        mask_low_inds = np.nonzero(mask_low < 1.0)
        if mask_low_inds[0].size > 0:
            spectra[freq_i, mask_low_inds] = 0

    masked_data = np.fft.ifft(np.fft.fftshift(spectra, axes=0), axis=0)
    masked_data = masked_data / window_expand

    # Unphase from zenith and cut to the desired band
    masked_data *= 1.0 / rephase_vals
    masked_data = masked_data.transpose(2, 0, 1)
    freq_ind_min = nfreq // 4
    freq_ind_max = 3 * nfreq // 4
    vis_model_arr = vis_model_arr[:, freq_ind_min:freq_ind_max]
    vis_model_arr[:, :, bi_use] = masked_data[:, freq_ind_min:freq_ind_max]

    return vis_model_arr

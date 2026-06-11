import logging
from pathlib import Path

from astropy import units
import numpy as np
from pyuvdata.utils.types import BoolArray, FloatArray
from pyradiosky import SkyModel

from ..beam_setup.beam_utils import beam_image
from ..pyfhd_tools.pyfhd_utils import angle_difference, region_grow, resistant_mean
from ..pyfhd_tools.unit_conv import pixel_to_radec, radec_to_pixel


def create_skymodel(
    *,
    obs: dict,
    psf: dict,
    logger: logging.Logger,
    skymodel: SkyModel | None = None,
    catalog_path: Path | str | None = None,
    sidelobe_catalog_path: Path | str | None = None,
    beam: FloatArray | None = None,
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
    logger : logging.Logger
        pyfhd's logger
    skymodel : SkyModel, optional
        SkyModel object to start with, if provided, catalog_path is ignored.
        Called "source_array" in FHD.
    catalog_path : Path or str, optional
        Path to the catalog to use.
    sidelobe_catalog_path : Path or str, optional
        Path to the catalog to use for the sidelobe sources if different than
        catalog to use for the main beam sources.
    beam : NDArray[np.float64], optional
        Average image-space beam per polarization. Created using `beam_image` if
        not provided.
    mask : NDArray[np.bool], optional
        Beam mask giving areas to include sources for (True where sources can be
        included, False where sources will be excluded).
    allow_sidelobe_sources : bool
        Include sidelobe sources. Also affects the defaulting of beam_threshold.
    beam_threshold : float
        Threshold for beam cut on sources. Sources belowthe beam_threshold will
        be cut from the skymodel to avoid sources in the nulls. Defaults to 0.05
        unless allow_sidelobe_sources is True, in which case the default is 0.01.
    restrict_sources : bool
        Option to restrict sources to near the beam center. Default is False.
        Related to "no_restrict_model_sources" and "no_restrict_cal_sources" but
        with the opposite sense to avoid double negatives.
    flux_threshold : float
        Threshold for flux values to include. These are catalog fluxes, not
        apparent (i.e. beam-weighted) fluxes. Can be negative, indicating an
        upper bound on fluxes.
    no_extend : bool
        Option to replace extended source components with a single component at
        the flux weighted average location with a flux equal to the total flux of
        all the components.
    max_sources : int
        Maximum number of sources to include, chosen from highest to lowest
        apparent (i.e. beam-weighted) flux. If a sidelobe_catalog_path is provided,
        sources are taken first from the main lobe catalog and then from the
        sidelobe catalog (if max_sources is greater than the number of sources
        in the main lobe catalog after the various cuts).
    spectral_index : float
        Spectral index to use for all sources. Overwrites the spectral index
        from the input catalog.
    preserve_zero_spectral_indices : bool
        Option to keep any spectral indices that are set to zero. Default is False,
        If False, the spectral index is reset to the mean spectral index of the
        catalog for any sources with zero spectral index.
    flatten_spectrum : bool
        Option to flatten the spectrum by the average spectral index (calculated
        as a flux-weighted average).
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
            "pyFHD currently only supports spectral index catalogs."
        )

    dimension = obs["dimension"]

    fov = (180 / np.pi) / obs["kpix"]

    freq_use = obs["freq_center"]
    n_pol = obs["n_pol"]

    if beam is None:
        # use at most xx & yy beams
        n_pol_use = np.min([n_pol, 2])
        beam = np.zeros((n_pol_use, dimension, dimension))

        for pol_i in range(n_pol_use):
            beam[pol_i] = beam_image(psf=psf, obs=obs, pol_i=pol_i)

        beam[np.nonzero(beam < 0)] = 0
    else:
        if (
            len(beam.shape) != 3
            or (beam.shape)[0] < np.min(n_pol, 2)
            or (beam.shape)[0] > np.max(n_pol, 2)
            or (beam.shape)[1:] != (dimension, dimension)
        ):
            raise ValueError("beam does not have expected shape.")

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

        sidelobe_skymodel = create_skymodel(
            obs=obs,
            psf=psf,
            logger=logger,
            catalog_path=sidelobe_catalog_path,
            beam=beam,
            mask=beam_sidelobe_mask,
            allow_sidelobe_sources=True,
            beam_threshold=beam_threshold,
            restrict_sources=restrict_sources,
            flux_threshold=flux_threshold,
            no_extend=no_extend,
            max_sources=max_sources,
            spectral_index=spectral_index,
            preserve_zero_spectral_indices=preserve_zero_spectral_indices,
        )
        allow_sidelobe_sources = False

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
                ext_src_lists[src] = wh_src

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
                for src, comp_arr in ext_src_lists.items():
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

                    assert skymodel.ra[first_comp] == ra
                    assert skymodel.dec[first_comp] == dec

                    keep_comp[comp_arr[1:]] = False

                skymodel.select(component_inds=np.nonzero(keep_comp)[0])
                assert np.all(skymodel.extended_model_group == "")
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
                    "All sources in model catalog after image and flux cuts are nan/inf."
                )
                skymodel = None
            else:
                if n_finite != skymodel.Ncomponents:
                    logger.warning(
                        "Model catalog contains nan/inf fluxes, dropping them."
                    )
                    skymodel.select(component_inds=inds_finite)
                    n_src_use = n_finite

                # if no extended sources survive the cuts, set extended_model_group to None
                if skymodel.extended_model_group is not None:
                    extended_comps = np.nonzero(skymodel.extended_model_group != "")[0]
                    if extended_comps.size == 0:
                        skymodel.extended_model_group = None

                # calculate apparent flux given the beam
                # NB: IDL FHD uses float indices into beam, which results in a
                # truncation of the float to an int. Here we use a round, but
                # it should be a very small difference (and only affects sorting
                # unless cutting on number of sources)
                influence = (
                    skymodel.extra_columns["flux_I_use"]
                    * beam[
                        skymodel.extra_columns["x_use"].astype(int),
                        skymodel.extra_columns["y_use"].astype(int),
                    ]
                )

                skymodel.add_extra_columns(
                    names=["beam_I"],
                    values=beam[
                        skymodel.extra_columns["x_use"].astype(int),
                        skymodel.extra_columns["y_use"].astype(int),
                    ],
                )

                # remove the extra columns just used internally
                skymodel.remove_extra_columns(
                    [
                        "ra_deg_use",
                        "dec_deg_use",
                        "x_use",
                        "y_use",
                        "flux_I_use",
                        "beam_I",
                    ]
                )

                # sort from max to min apparent flux
                order = np.flip(np.argsort(influence))
                skymodel._select_along_param_axis({"Ncomponents": order})

    if sidelobe_catalog_path is not None:
        if skymodel is not None:
            skymodel.concat(sidelobe_skymodel)
        else:
            logger.warning(
                "No remaining sources in model catalog after cuts, only using sidelobe catalog."
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
                extended_srcs = np.unique(
                    skymodel.extended_model_group[extended_comps], return_index=True
                )[1]

                ext_src_lists = {}
                extended_start_inds = np.zeros_like(extended_srcs)
                for src_i, src in enumerate(extended_srcs):
                    wh_src = np.nonzero(skymodel.extended_model_group == src)[0]
                    ext_src_lists[src] = wh_src
                    extended_start_inds[src_i] = wh_src

                src_inds_select = np.sort(
                    np.concatenate(compact_src_inds, extended_start_inds)
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

                comp_keep = np.concatenate(compact_keep, extended_keep)
        skymodel.select(component_inds=comp_keep)

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

    return skymodel

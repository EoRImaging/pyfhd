import numpy as np
from astropy.constants import c
from pyuvdata import BeamInterface, UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from PyFHD.pyfhd_tools.pyfhd_utils import histogram, region_grow
import h5py
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates


def gaussian_decomp(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    ftransform: bool = False,
    model_npix: float | None = None,
    model_res: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Create an analytically built Gaussian decomposition of the beam on
    the supplied x-y grid given the input Gaussian parameters. If
    ftransform is True, then the analytic Fourier Transform of the input
    gaussians on the supplied x-y grid is returned. Any number of Gaussians
    can be supplied in the p vector. To transfer Gaussian parameters from
    a different grid to the current x-y grid, the model_npix and model_res
    parameters can be supplied.

    Parameters
    ----------
    x : np.ndarray
        X-axis vector of pixel numbers
    y : np.ndarray
        Y-axis vector of pixel numbers
    p : np.ndarray
        Gaussian parameter vector, ordered as amp, offset x, sigma x, offset y,
        sigma y per lobe
    ftransform : bool, optional
        Return the analytic Fourier Transform of the input gaussians on the supplied
        x-y grid, by default False
    model_npix : float | None, optional
        The number of pixels on an axis used to derive the input parameters to
        convert to the current x-y grid, by default None
    model_res : float | None, optional
        The grid resolution used to derive the input parameters to convert to
        the current grid resolution, by default None

    Returns
    -------
    tuple[np.ndarray, float, float]
        The Gaussian decomposition of the beam on the supplied x-y grid given the input
        Gaussian parameters, along with the analytic volume and squared volume of the
        Gaussian decomposition.
    """
    decomp_beam = np.zeros([x.size, y.size])
    i = 1j
    # Expand the p vector into readable names
    var = np.reshape(p, [p.size // 5, 5])
    amp = var[:, 0]
    offset_x = var[:, 1]
    sigma_x = var[:, 2]
    offset_y = var[:, 3]
    sigma_y = var[:, 4]
    n_lobes = var[:, 0].size

    # If the parameters were built on a different grid, then put on new grid
    # Npix only affects the offset params
    if model_npix is not None:
        if model_npix < x.size:
            offset = np.abs(x.size / 2 - model_npix / 2)
            offset_x += offset
            offset_y += offset
        else:
            offset = np.abs(model_npix / 2 - x.size / 2)
            offset_x -= offset
            offset_y -= offset
    # Resolution affects gaussian sigma and offsets
    if model_res is not None:
        sigma_x *= model_res
        sigma_y *= model_res
        offset_x = ((offset_x - x.size / 2) * model_res) + x.size / 2
        offset_y = ((offset_y - y.size / 2) * model_res) + y.size / 2

    if not ftransform:
        for lobe in range(n_lobes):
            decomp_beam += amp[lobe] * np.outer(
                np.exp(-((y - offset_y[lobe]) ** 2) / (2 * sigma_y[lobe] ** 2)),
                np.exp(-((x - offset_x[lobe]) ** 2) / (2 * sigma_x[lobe] ** 2)),
            )
        volume_beam = 0
        sq_volume_beam = 0
    else:
        # Full uv model with all the gaussian components
        decomp_beam = decomp_beam.astype(np.complex128)
        volume_beam = np.sum(amp)
        sq_volume_beam = np.pi * np.sum(sigma_x * sigma_y * amp**2) / (x.size * y.size)

        offset_x -= x.size / 2
        offset_y -= y.size / 2

        kx = np.outer(np.arange(x.size) - x.size / 2, np.ones(y.size))
        ky = np.outer(np.ones(x.size), np.arange(y.size) - y.size / 2)
        decomp_beam += (
            amp**2
            * np.pi
            / (x.size * y.size)
            * sigma_x
            * sigma_y
            * np.exp(
                (
                    2 * np.pi**2 / (x.size * y.size) * sigma_x**2 * kx**2
                    + sigma_y**2 * ky**2
                )
                - (2 * np.pi / x.size * 1j * (offset_x * kx + offset_y * ky))
            )
        )

    return decomp_beam, volume_beam, sq_volume_beam


def beam_image(
    psf: dict | h5py.File,
    obs: dict,
    pol_i: int,
    freq_i: int | None = None,
    abs=False,
    square=False,
) -> np.ndarray:
    """
    Generates the average beam in image space for one polarization over all
    frequencies, or optionally for one frequency. The UV->sky transformation
    uses the inverse FFT for the beam, but the forward FFT for the image.
    This convention ensures the correct orientation of the UV-space beam
    for gridding visibilities. If the psf dictionary has Gaussian parameters,
    then the Gaussian decomposition is used to generate the analytic beam image.

    Parameters
    ----------
    psf : dict
        Beam dictionary
    obs : dict
        Observation metadata dictionary
    pol_i : int
        Index of the polarization to use
    freq_i : int
        Index of the frequency to use, by default None
    abs : bool, optional
        Return the absolute value of the beam image, by default False
    square : bool, optional
        Return the square of the beam image, by default False

    Returns
    -------
    beam_base : np.ndarray
        The average beam in image space for the specified polarization
        and all frequencies, or for a specific frequency if freq_i is set.
    """

    psf_dim = psf["dim"]
    if "freq_norm" in psf:
        freq_norm = psf["freq_norm"]
    elif "fnorm" in psf:
        # handling for older files or imports from IDL FHD
        freq_norm = psf["fnorm"]
    pix_horizon = psf["pix_horizon"]
    group_id = psf["id"][pol_i, 0, :]
    if "beam_gaussian_params" in psf:
        beam_gaussian_params = psf["beam_gaussian_params"][:]
    else:
        beam_gaussian_params = None
    rbin = 0
    # If we lazy loaded psf, get actual numbers out of the datasets
    if isinstance(psf, h5py.File):
        psf_dim = psf_dim[0]
        freq_norm = freq_norm[:]
        pix_horizon = pix_horizon[0]
    dimension = elements = obs["dimension"]
    # these should all be integers b/c dimensions are usually even numbers.
    # but they have to be cast to ints to be used in slicing.
    xl = int(dimension / 2 - psf_dim / 2 + 1)
    xh = int(dimension / 2 - psf_dim / 2 + psf_dim)
    yl = int(elements / 2 - psf_dim / 2 + 1)
    yh = int(elements / 2 - psf_dim / 2 + psf_dim)

    group_n, _, ri_id = histogram(group_id, min=0)
    gi_use = np.nonzero(group_n)
    # Most likely going to be 1 as PyFHD does only one beam mostly
    n_groups = np.count_nonzero(group_n)

    if beam_gaussian_params is not None:
        # 1.3 is the padding factor for the gaussian fitting procedure
        # (2.*obs.kpix) is the ratio of full sky (2 in l,m) to the analysis range (1/obs.kpix)
        # (2.*obs.kpix*dimension/psf.pix_horizon) is the scale factor between the psf pixels-to-horizon and the
        # analysis pixels-to-horizon
        # (0.5/obs.kpix) is the resolution scaling of what the beam model was made at and the current res
        model_npix = pix_horizon * 1.3
        model_res = (2 * obs["kpix"] * dimension) / pix_horizon * (0.5 / obs["kpix"])

    freq_bin_i = obs["baseline_info"]["fbin_i"]
    freq_i_use = np.nonzero(obs["baseline_info"]["freq_use"])[0]
    n_bin_use = 0
    # We assume freq_i is an int when provided (i.e. a single frequency index)
    if freq_i is not None:
        freq_i_use = freq_i

    if square:
        # Do note freq_i_use could be an integer or an array if freq_i is supplied or not
        beam_base = np.zeros([dimension, elements])
        freq_bin_use = freq_bin_i[freq_i_use]
        fbin_use = np.sort(np.unique(freq_bin_use))
        nbin = fbin_use.size

        if beam_gaussian_params is not None:
            beam_single = np.zeros([dimension, elements])
        else:
            beam_single = np.zeros([psf_dim, psf_dim], dtype=np.complex128)
        for bin_i in range(nbin):
            fbin = fbin_use[bin_i]
            nf_bin = np.count_nonzero(freq_bin_use == fbin)
            if beam_gaussian_params is not None:
                for gi in range(n_groups):
                    # beam_gaussian_params needs to be copied here rather than
                    # a view as interestingly gaussian_decomp affects the values
                    # of the array used with the calculations done to var and no copies
                    # are made, only views are adjusted. so we explcitly call copy
                    params = beam_gaussian_params[pol_i, fbin, :].copy()
                    gaussian = gaussian_decomp(
                        np.arange(dimension),
                        np.arange(elements),
                        params,
                        model_npix=model_npix,
                        model_res=model_res,
                    )[0]
                    beam_single += gaussian * group_n[gi_use[gi]]
                beam_single /= np.sum(group_n[gi_use])
                beam_base += nf_bin * beam_single**2
            else:
                for gi in range(n_groups):
                    beam_single += (
                        psf["beam_ptr"][0, fbin, rbin, rbin] * group_n[gi_use[gi]]
                    ).reshape([psf_dim, psf_dim])
                beam_single /= np.sum(group_n[gi_use])
                if abs:
                    beam_single = np.abs(beam_single)
                beam_base_uv1 = np.zeros([dimension, elements], np.complex128)
                beam_base_uv1[xl : xh + 1, yl : yh + 1] = beam_single
                beam_base_single = np.fft.fftshift(
                    np.fft.ifftn(np.fft.fftshift(beam_base_uv1))
                )
                beam_base += (
                    nf_bin * (beam_base_single * np.conjugate(beam_base_single)).real
                )
            n_bin_use += nf_bin * freq_norm[fbin]
    else:
        nf_use = freq_i_use.size
        if beam_gaussian_params is not None:
            beam_base_uv = np.zeros([dimension, elements])
            beam_single = np.zeros([dimension, elements])
        else:
            beam_base_uv = np.zeros([psf_dim, psf_dim], dtype=np.complex128)
            beam_single = np.zeros([psf_dim, psf_dim], dtype=np.complex128)
        for f_idx in range(nf_use):
            fi = freq_i_use[f_idx]
            if freq_i is not None:
                if freq_i != fi:
                    continue
            fbin = freq_bin_i[fi]
            beam_single[:, :] = 0
            if beam_gaussian_params is not None:
                for gi in range(n_groups):
                    params = beam_gaussian_params[pol_i, fbin, :].copy()
                    gaussian = gaussian_decomp(
                        np.arange(dimension),
                        np.arange(elements),
                        params,
                        model_npix=model_npix,
                        model_res=model_res,
                    )[0]
                    beam_single += gaussian * group_n[gi_use[gi]]
            else:
                for gi in range(n_groups):
                    beam_single += (
                        psf["beam_ptr"][0, fbin, rbin, rbin] * group_n[gi_use[gi]]
                    ).reshape([psf_dim, psf_dim])
            beam_single /= np.sum(group_n[gi_use])
            beam_base_uv += beam_single
            n_bin_use += freq_norm[fbin]

        if beam_gaussian_params is None:
            beam_base_uv1 = np.zeros([dimension, elements], dtype=np.complex128)
            beam_base_uv1[xl : xh + 1, yl : yh + 1] = beam_base_uv
            beam_base = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(beam_base_uv1)))
        else:
            beam_base = beam_base_uv

    beam_base /= n_bin_use
    return beam_base.real


def beam_image_hyperresolved(
    antenna: dict,
    beam: UVBeam | AnalyticBeam,
    ant_pol_1: int,
    ant_pol_2: int,
    freq_i: int,
    zen_int_x: float,
    zen_int_y: float,
    psf: dict,
) -> NDArray[np.complexfloating]:
    """
    Build the hyperresolved image-space beam power for a station/tile. Currently, this
    function assumes that the amplitude of the jones matrix response between two antennas,
    multiplied by the station/tile response, is the image beam power. This calculation
    may be suject to change in the future.

    Parameters
    ----------
    antenna : dict
        Antenna metadata dictionary
    beam : UVBeam | AnalyticBeam
        UVBeam or AnalyticBeam object containing the beam model and metadata.
    ant_pol_1 : int
        Polarization index for the first antenna
    ant_pol_2 : int
        Polarization index for the second antenna
    freq_i : int
        Frequency index for current iteration
    zen_int_x : float
        x pixel index of the zenith for the beam power image
    zen_int_y : float
        y pixel index of the zenith for the beam power image
    psf : dict
        Beam metadata dictionary

    Returns
    -------
    NDArray[np.complexfloating]
        An estimation of the image-space beam power, normalized to the zenith power.
    """
    # FHD was designed to account for multiple antennas but in most cases only one was ever used
    # So we will just use the first antenna twice as I PyFHD does not support multiple antennas at this time,
    # If you want to use multiple antennas, please open an issue on the PyFHD GitHub repository or do the translation and/or
    # adjustments yourself.
    beam_ant_1 = antenna["response"][ant_pol_1, freq_i]
    beam_ant_2 = np.conjugate(antenna["response"][ant_pol_2, freq_i])

    # Amplitude of the response from "ant1" (again FHD takes more than one antenna)
    # is Sqrt(|J1[0,pol1]|^2 + |J1[1,pol1]|^2)
    amp_1 = (
        np.abs(antenna["jones"][0, ant_pol_1]) ** 2
        + np.abs(antenna["jones"][1, ant_pol_1]) ** 2
    )
    # Amplitude of the response from "ant2" (again FHD takes more than one antenna)
    # is Sqrt(|J2[0,pol2]|^2 + |J2[1,pol2]|^2)
    amp_2 = (
        np.abs(antenna["jones"][0, ant_pol_2]) ** 2
        + np.abs(antenna["jones"][1, ant_pol_2]) ** 2
    )
    # Amplitude of the baseline response is the product of the "two" antenna responses
    power_zenith_beam = np.sqrt(amp_1 * amp_2)

    # Create one full-scale array
    image_power_beam = np.zeros([psf["dim"], psf["dim"]], dtype=np.complex128)

    # Co-opt the array to calculate the power at zenith
    image_power_beam[antenna["pix_use"]] = power_zenith_beam
    # TODO: Work out the interpolation of the zenith power, it uses cubic interpolation
    # But the IDL Interpolate function in IDL uses an interpolation paramter of -0.5, where
    # scipy, numpy with their B-Splines seem to use a parameter of 0 by default with no way
    # to change it.
    # The interp is a placeholder for now, but it should be replaced with a proper
    # interpolation function that matches the IDL Interpolate function.
    # TODO: Replace with UVBeam interface object as_power_beam, then
    # compute_response at za 0 az 0
    # Initial trying out of using pyuvdata, not close at all. This is interpolating
    # the zenith power using the x and y pixel coordinates, to use pyuvdata likely need to do
    # pixel to ra/dec then to za/az
    power_zenith = np.interp(zen_int_x, zen_int_y, image_power_beam)

    # Normalize the image power beam to the zenith
    image_power_beam[antenna["pix_use"]] = (
        power_zenith_beam * beam_ant_1 * beam_ant_2
    ) / power_zenith

    return image_power_beam


def beam_power(
    antenna: dict,
    beam: UVBeam | AnalyticBeam,
    ant_pol_1: int,
    ant_pol_2: int,
    freq_i: int,
    psf: dict,
    zen_int_x: float,
    zen_int_y: float,
    xvals_uv_superres: np.ndarray,
    yvals_uv_superres: np.ndarray,
    pyfhd_config: dict,
) -> NDArray[np.complexfloating]:
    """
    Generate the hyperresolved image-space beam power to reduce aliasing artifacts, and
    fourier transform it to a specific grid in complex uv-space. Reduce artifacts further
    by applying an extremely low-level contiguous mask to the uv-space beam power and
    renomalizing the beam power to a volume of 1.

    Parameters
    ----------
    antenna : dict
        Antenna metadata dictionary
    beam : UVBeam | AnalyticBeam
       UVBeam or AnalyticBeam object containing the beam model and metadata.
    ant_pol_1 : int
        Polarization index for the first antenna
    ant_pol_2 : int
        Polarization index for the second antenna
    freq_i : int
        Frequency index for current iteration
    psf : dict
        Beam metadata dictionary
    zen_int_x : np.ndarray
        x pixel index of the zenith for the beam power image
    zen_int_y : np.ndarray
        y pixel index of the zenith for the beam power image
    xvals_uv_superres : np.ndarray
        A grid of the hyperresolved indices in the u direction for the uv-space beam power image
    yvals_uv_superres : np.ndarray
        A grid of the hyperresolved indices in the v direction for the uv-space beam power image
    pyfhd_config : dict
        The PyFHD configuration dictionary

    Returns
    -------
    NDArray[np.complexfloating]
        Hyperresolved uv-space of the beam power image, normalized to a volume of 1
    """
    # For now we will ignore beam_gaussian_decomp and much of the debug keywords
    image_power_beam = beam_image_hyperresolved(
        antenna,
        beam,
        ant_pol_1,
        ant_pol_2,
        freq_i,
        zen_int_x,
        zen_int_y,
        psf,
        pyfhd_config,
    )
    if pyfhd_config["kernel_window"]:
        image_power_beam *= antenna["pix_window"]
    psf_base_single = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(image_power_beam)))
    # TODO: Same cubic problem as in beam_image_hyperresolved here
    # Map Coordinates isn't the same as IDL Interpolate
    # But its closish, more of a placeholder for now.
    psf_base_superres = map_coordinates(
        psf_base_single, (xvals_uv_superres, yvals_uv_superres)
    )

    # Build a mask to create a well-defined finite beam
    uv_mask_superres = np.zeros(
        [[psf_base_superres.shape[0], psf_base_superres.shape[1]]], dtype=np.float64
    )
    psf_mask_threshold_use = (
        np.max(np.abs(psf_base_superres)) / pyfhd_config["beam_mask_threshold"]
    )
    beam_i = region_grow(
        np.abs(psf_base_superres),
        psf["superres_dim"] * (1 + psf["superres_dim"]) / 2,
        low=psf_mask_threshold_use,
        high=np.max(np.abs(psf_base_superres)),
    )
    uv_mask_superres[beam_i] = 1

    # FFT normalization correction in case this changes the total number of pixels
    psf_base_superres *= psf["intermediate_res"] ** 2

    """
    total of the gaussian decomposition can be calculated analytically, but is an over-estimate 
    of the numerical representation and results in a beam norm of greater than one,
    thus the discrete total is used
    """
    psf_val_ref = np.sum(psf_base_superres)

    # If you wish to add interpolate_beam_threshold functionality then do so here
    psf_base_superres *= uv_mask_superres

    if pyfhd_config["beam_clip_floor"]:
        i_use = np.where(np.abs(psf_base_superres))
        psf_amp = np.abs(psf_base_superres)
        psf_phase = np.arctan(psf_base_superres.imag / psf_base_superres.real)
        psf_floor = psf_mask_threshold_use * (psf["intermediate_res"] ** 2)
        psf_amp[i_use] -= psf_floor
        psf_base_superres = psf_amp * np.cos(psf_phase) + 1j * psf_amp * np.sin(
            psf_phase
        )

    psf_base_superres *= psf_val_ref / np.sum(psf_base_superres)

    return psf_base_superres

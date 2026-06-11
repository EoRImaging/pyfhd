import numpy as np
import pytest
from pyradiosky import SkyModel

from pyfhd.data.datasets import fetch_data
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.pyfhd_tools.test_utils import get_savs, sav_file_rearrange_psf
from pyfhd.source_modeling.source_utils import create_skymodel


@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
@pytest.mark.parametrize("refraction", [None, "astropy", "idl"])
def test_create_skymodel(refraction):

    catalog_path = fetch_data("gleam_rlb2019_cut")

    obs_file = fetch_data("2013_zenith_obs")
    psf_cut_file = fetch_data("2013_zenith_psf_small")

    obs_sav_dict = get_savs(obs_file, "")
    obs_sav_dict = recarray_to_dict(obs_sav_dict)
    obs = obs_sav_dict["obs"]
    psf_sav_dict = get_savs(psf_cut_file, "")
    # fix the psf to be properly arranged
    psf = sav_file_rearrange_psf(psf_sav_dict["psf"])

    idl_cal_sources_file = fetch_data("2013_zenith_gleam_sources")

    expected_sky = SkyModel.from_file(
        idl_cal_sources_file, extra_columns={"x": "image_x", "y": "image_y"}
    )

    expected_sky.at_frequencies(np.atleast_1d(expected_sky.reference_frequency[0]))

    # obs dimension should be an int
    obs["dimension"] = int(obs["dimension"])

    # freqs = np.array([1.6512e08, 1.8048e08])
    # set obs freq_array to match psf
    freqs = psf["freq"]
    n_freq = freqs.size
    obs["n_freq"] = n_freq
    obs["nf_vis"] = obs["nf_vis"][0:n_freq, :]
    obs["freq_center"] = np.mean(freqs)
    obs["baseline_info"]["freq"] = freqs
    obs["baseline_info"]["freq_use"] = np.ones((n_freq,), dtype=int)
    obs["baseline_info"]["fbin_i"] = np.arange(n_freq, dtype=int)

    sky = create_skymodel(
        obs=obs, psf=psf, catalog_path=catalog_path, logger=None, refraction=refraction
    )

    if refraction != "idl":
        # sorting is slightly different because of different beam values with refraction
        sky._select_along_param_axis(
            {"Ncomponents": np.flip(np.argsort(sky.stokes[0, 0, :]))}
        )
        expected_sky._select_along_param_axis(
            {"Ncomponents": np.flip(np.argsort(expected_sky.stokes[0, 0, :]))}
        )

    if refraction is None:
        image_xy_atol = 0.09
    elif refraction == "astropy":
        image_xy_atol = 0.04
    else:
        image_xy_atol = 0.0011

    for col_name in ["image_x", "image_y"]:
        np.testing.assert_allclose(
            sky.extra_columns[col_name],
            expected_sky.extra_columns[col_name],
            rtol=0,
            atol=image_xy_atol,
        )

    # remove extra cols to test the rest of the objects
    sky.extra_columns = None
    expected_sky.extra_columns = None

    # FHD renumbers the components, which mucks up the name attribute
    # set them equal to enable comparison
    sky.name = expected_sky.name

    # make histories match for comparison purposes
    sky.history = expected_sky.history

    assert sky == expected_sky

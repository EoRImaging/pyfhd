import copy

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Longitude, Latitude
from pyradiosky import SkyModel

from pyfhd.data.datasets import fetch_data
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.source_modeling.source_utils import (
    generate_source_cal_skymodel,
    stokes_cnv,
    source_dft,
    vis_delay_filter,
    _setup_uv_vals,
)
from pyfhd.source_modeling.vis_source_model import vis_source_model
from pyfhd.pyfhd_tools.test_utils import get_savs
from pyfhd.pyfhd_tools.unit_conv import altaz_to_radec, radec_to_pixel


@pytest.fixture(scope="session")
def gleam_cut_2013_sky(mwa_aee_beam_zenith_2013):

    _, psf, obs, _ = mwa_aee_beam_zenith_2013
    catalog_path = fetch_data("gleam_rlb2019_cut")

    sky = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=catalog_path)

    yield sky


@pytest.fixture(scope="session")
def fixed_obs(zenith_obs_2013_main, zenith_psf_2013_cut):
    psf = zenith_psf_2013_cut
    obs = copy.deepcopy(zenith_obs_2013_main)

    # obs dimension should be an int
    obs["dimension"] = int(obs["dimension"])
    # center pixel should be an int
    obs["obsx"] = int(obs["obsx"])
    obs["obsy"] = int(obs["obsy"])

    # set obs freq_array to match psf
    freqs = psf["freq"]
    n_freq = freqs.size
    obs["n_freq"] = n_freq
    obs["nf_vis"] = obs["nf_vis"][0:n_freq]
    obs["freq_center"] = np.mean(freqs)
    obs["baseline_info"]["freq"] = freqs
    obs["baseline_info"]["freq_use"] = np.ones((n_freq,), dtype=int)
    obs["baseline_info"]["fbin_i"] = np.arange(n_freq, dtype=int)

    yield obs


def split_source_as_extended(filein, name, fileout):
    ext_sky = SkyModel.from_file(filein)

    ind = np.nonzero(ext_sky.name == name)[0][0]
    ext_model = [""] * ext_sky.Ncomponents
    ext_model[ind] = "test_ext"
    assert ext_model[ind] == "test_ext"
    ext_sky.extended_model_group = np.array(ext_model)
    ext_sky.stokes[0, 0, ind] = ext_sky.stokes[0, 0, ind] / 2.0

    new_sky = ext_sky.select(component_inds=np.atleast_1d([ind]), inplace=False)
    new_sky.name[0] += "a"
    ext_sky.concat(new_sky)

    ext_sky.write_skyh5(fileout)


@pytest.fixture(scope="session")
def split_kept_source(tmp_path_factory):
    catalog_path = fetch_data("gleam_rlb2019_cut")

    # make a copy but divide one source into two extended components
    # chosen as one that is selected
    name = "12082"
    ext_cat_name = (
        tmp_path_factory.mktemp("src_utils_split") / "gleam_rlb2019_cut_split_ext.skyh5"
    )
    split_source_as_extended(filein=catalog_path, name=name, fileout=ext_cat_name)

    yield ext_cat_name


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
@pytest.mark.parametrize("refraction", [None, "astropy", "idl"])
def test_gen_cal_sky(fixed_obs, zenith_psf_2013_cut, refraction):
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    catalog_path = fetch_data("gleam_rlb2019_cut")

    fhd_cal_sources_file = fetch_data("2013_zenith_gleam_sources")

    expected_sky = SkyModel.from_file(
        fhd_cal_sources_file, extra_columns={"x": "image_x", "y": "image_y"}
    )

    # take this out once pyradiosky fix is in:
    if expected_sky.extended_model_group is not None:
        extended_comps = np.nonzero(expected_sky.extended_model_group != "")[0]
        if extended_comps.size == 0:
            expected_sky.extended_model_group = None

    expected_sky.at_frequencies(np.atleast_1d(expected_sky.reference_frequency[0]))

    sky = generate_source_cal_skymodel(
        obs=obs, psf=psf, catalog_path=catalog_path, refraction=refraction
    )

    # sorting is slightly different because of different beam values
    # from round vs truncate pixel and refraction (if refraction != "idl")
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
        # even when refraction matches, the round vs truncation causes a difference
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


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
@pytest.mark.parametrize("beam_thresh", [None, 0.02])
def test_gen_cal_sky_sidelobes(fixed_obs, zenith_psf_2013_cut, beam_thresh):
    """
    Check that passing the same catalog as the sidelobe catalog gives the same
    answer as not passing a separate sidelobe catalog.
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    catalog_path = fetch_data("gleam_rlb2019_cut")

    sky_single_cat = generate_source_cal_skymodel(
        obs=obs,
        psf=psf,
        catalog_path=catalog_path,
        allow_sidelobe_sources=True,
        beam_threshold=beam_thresh,
    )

    sky_two_cat = generate_source_cal_skymodel(
        obs=obs,
        psf=psf,
        catalog_path=catalog_path,
        allow_sidelobe_sources=True,
        sidelobe_catalog_path=catalog_path,
        beam_threshold=beam_thresh,
    )

    # sorting is slightly different
    sky_single_cat._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(sky_single_cat.stokes[0, 0, :]))}
    )
    sky_two_cat._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(sky_two_cat.stokes[0, 0, :]))}
    )

    # make histories match for comparison purposes
    sky_two_cat.history = sky_single_cat.history

    assert sky_single_cat == sky_two_cat


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_gen_cal_sky_icrs(fixed_obs, zenith_psf_2013_cut, tmpdir):
    """
    Check that skymodel frames get transformed properly
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    catalog_path = fetch_data("gleam_rlb2019_cut")

    # make a copy but transform to icrs
    new_sky = SkyModel.from_file(catalog_path)
    new_sky.transform_to("icrs")
    new_cat_name = tmpdir / "gleam_rlb2019_cut_icrs.skyh5"
    new_sky.write_skyh5(new_cat_name)

    sky1 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=catalog_path)

    sky2 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=new_cat_name)

    # make histories match for comparison purposes
    sky2.history = sky1.history

    assert sky1 == sky2


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_gen_cal_sky_flux_thresh(fixed_obs, zenith_psf_2013_cut, tmpdir):
    """
    Check that flux_thresholding is done correctly
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    catalog_path = fetch_data("gleam_rlb2019_cut")
    flux_thresh = 1.0  # Jy

    # make a copy but set cut on flux
    new_sky = SkyModel.from_file(catalog_path)
    comp_inds = np.nonzero(new_sky.stokes[0, 0] > flux_thresh * units.Jy)
    new_sky.select(component_inds=comp_inds[0])
    new_cat_name = tmpdir / "gleam_rlb2019_cut_bright.skyh5"
    new_sky.write_skyh5(new_cat_name)

    sky1 = generate_source_cal_skymodel(
        obs=obs, psf=psf, catalog_path=catalog_path, flux_threshold=flux_thresh
    )

    sky2 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=new_cat_name)

    # make histories match for comparison purposes
    sky2.history = sky1.history

    assert sky1 == sky2


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
@pytest.mark.parametrize("extended", [False, True])
def test_gen_cal_sky_max_src(
    fixed_obs, zenith_psf_2013_cut, extended, split_kept_source
):
    """
    Check that selecting a maximum number of sources works properly
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    if extended:
        catalog_path = split_kept_source
    else:
        catalog_path = fetch_data("gleam_rlb2019_cut")
    src_cap = 100

    sky1 = generate_source_cal_skymodel(
        obs=obs, psf=psf, catalog_path=catalog_path, max_sources=src_cap
    )

    sky2 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=catalog_path)
    if extended:
        src_cap += 1

    sky2.select(component_inds=np.arange(src_cap))

    # make histories match for comparison purposes
    sky2.history = sky1.history

    assert sky1 == sky2


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_gen_cal_sky_extended(fixed_obs, zenith_psf_2013_cut, tmpdir):
    """
    Check that extended sources are handled properly
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    catalog_path = fetch_data("gleam_rlb2019_cut")

    # make a copy but set some sources as extended
    # chosen as ones that are selected and fairly close together
    names = ["12082", "12771"]
    ext_sky = SkyModel.from_file(catalog_path)
    ext_model = [""] * ext_sky.Ncomponents
    for name in names:
        ind = np.nonzero(ext_sky.name == name)[0][0]
        ext_model[ind] = "test_ext"
        assert ext_model[ind] == "test_ext"
    ext_sky.extended_model_group = np.array(ext_model)
    ext_cat_name = tmpdir / "gleam_rlb2019_cut_ext.skyh5"
    ext_sky.write_skyh5(ext_cat_name)

    sky1 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=catalog_path)

    sky2 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=ext_cat_name)
    assert sky2.extended_model_group is not None
    np.testing.assert_array_equal(
        np.unique(sky2.extended_model_group), np.array(["", "test_ext"])
    )
    np.testing.assert_array_equal(
        np.sort(sky2.name[np.nonzero(sky2.extended_model_group == "test_ext")]),
        np.sort(np.array(names)),
    )
    # remove extended model group for comparison
    sky2.extended_model_group = None

    # make beam values match
    for name in names:
        ind1 = np.nonzero(sky1.name == name)[0][0]
        ind2 = np.nonzero(sky2.name == name)[0][0]
        assert sky2.extra_columns["beam_I"][ind2] != sky1.extra_columns["beam_I"][ind1]
        sky2.extra_columns["beam_I"][ind2] = sky1.extra_columns["beam_I"][ind1]

    # sorting is slightly different
    sky1._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(sky1.stokes[0, 0, :]))}
    )
    sky2._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(sky2.stokes[0, 0, :]))}
    )

    # make histories match for comparison purposes
    sky2.history = sky1.history

    assert sky1 == sky2


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
@pytest.mark.parametrize("restrict_sources", [True, False])
def test_gen_cal_sky_extend_collapse(
    fixed_obs, zenith_psf_2013_cut, split_kept_source, restrict_sources
):
    """
    Check that collapsing extended sources is handled properly
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    catalog_path = fetch_data("gleam_rlb2019_cut")
    ext_cat_path = split_kept_source

    sky1 = generate_source_cal_skymodel(
        obs=obs, psf=psf, catalog_path=catalog_path, restrict_sources=restrict_sources
    )

    sky2 = generate_source_cal_skymodel(
        obs=obs,
        psf=psf,
        catalog_path=ext_cat_path,
        no_extend=True,
        restrict_sources=restrict_sources,
    )

    # make histories match for comparison purposes
    sky2.history = sky1.history

    assert sky1 == sky2


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_gen_cal_sky_extend_cut(fixed_obs, zenith_psf_2013_cut, tmpdir):
    """
    Check handling if all extended sources are cut.
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs

    catalog_path = fetch_data("gleam_rlb2019_cut")
    outfile = tmpdir / "gleam_rlb2019_cut_ext_cut.skyh5"
    split_source_as_extended(filein=catalog_path, name="100224", fileout=outfile)

    sky1 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=catalog_path)

    sky2 = generate_source_cal_skymodel(obs=obs, psf=psf, catalog_path=outfile)
    assert sky2.extended_model_group is None

    # sorting is slightly different
    sky1._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(sky1.stokes[0, 0, :]))}
    )
    sky2._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(sky2.stokes[0, 0, :]))}
    )

    # make histories match for comparison purposes
    sky2.history = sky1.history

    assert sky1 == sky2


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
@pytest.mark.parametrize(
    ("err_type", "message", "kwargs"),
    [
        (ValueError, "catalog_path does not exist:", {"catalog_path": "foo"}),
        (
            TypeError,
            "If provided, skymodel must be a pyradiosky.SkyModel object.",
            {"skymodel": "foo", "catalog_path": None},
        ),
        (
            NotImplementedError,
            "pyfhd currently only supports spectral index catalogs.",
            None,
        ),
        (ValueError, "beam does not have expected shape.", {"beam": np.arange(10)}),
        (TypeError, "beam must be an array.", {"beam": "foo"}),
    ],
)
def test_gen_cal_sky_errors(fixed_obs, zenith_psf_2013_cut, err_type, message, kwargs):
    """
    Check that collapsing extended sources is handled properly
    """
    psf = zenith_psf_2013_cut
    obs = fixed_obs
    catalog_path = fetch_data("gleam_rlb2019_cut")

    kwargs_use = {"obs": obs, "psf": psf, "catalog_path": catalog_path}

    if kwargs is None:
        sky = SkyModel.from_file(catalog_path)
        sky.at_frequencies(np.atleast_1d(obs["freq_center"]) * units.Hz)
        kwargs = {"skymodel": sky, "catalog_path": None}

    kwargs_use.update(kwargs)

    with pytest.raises(err_type, match=message):
        generate_source_cal_skymodel(**kwargs_use)


@pytest.mark.github_actions
def test_stokes_cnv_zenith_src(mwa_aee_beam_zenith_2013):
    antenna, _, obs, _ = mwa_aee_beam_zenith_2013

    n_comp = 3
    zenith_sky = SkyModel(
        name=[f"zen_src_{ind}" for ind in range(n_comp)],
        ra=Longitude([obs["zenra"] * units.deg] * n_comp),
        dec=Latitude([obs["zendec"] * units.deg] * n_comp),
        spectral_type="spectral_index",
        reference_frequency=np.array([obs["freq_center"]] * n_comp) * units.Hz,
        spectral_index=[-0.8] * n_comp,
        stokes=np.repeat(np.reshape([1, 0, 0, 0], (4, 1, 1)), n_comp, axis=2)
        * units.Jy,
        frame="fk5",
        extra_column_dict={
            "image_x": np.array([obs["zenx"]] * n_comp),
            "image_y": np.array([obs["zeny"]] * n_comp),
        },
    )
    new_sky = stokes_cnv(sky=zenith_sky, antenna=antenna, obs=obs, inverse=True)

    # I expected a slightly better tolerance than this...
    for pol_i in range(2):
        np.testing.assert_allclose(
            new_sky.extra_columns[f"flux_pol_{pol_i}"],
            np.full((n_comp,), 0.5, dtype=np.complex128),
            atol=2e-5,
            rtol=0,
        )


@pytest.mark.github_actions
def test_stokes_cnv_off_zenith(mwa_aee_beam_zenith_2013):
    antenna, _, obs, _ = mwa_aee_beam_zenith_2013

    za_grid = np.linspace(5, 15, 5)
    az_grid = np.linspace(0, 360, 8, endpoint=False)

    za_array, az_array = np.meshgrid(za_grid, az_grid)
    za_array = za_array.flatten()
    az_array = az_array.flatten()

    ra_array, dec_array = altaz_to_radec(
        alt=90 - za_array,
        az=az_array,
        lat=obs["lat"],
        lon=obs["lon"],
        height=obs["alt"],
        time=obs["jd0"],
    )

    image_x, image_y = radec_to_pixel(ra_array, dec_array, obs["astr"])

    n_comp = za_array.size
    sky = SkyModel(
        name=[f"grid_src_{ind}" for ind in range(n_comp)],
        ra=Longitude(ra_array * units.deg),
        dec=Latitude(dec_array * units.deg),
        spectral_type="spectral_index",
        reference_frequency=np.array([obs["freq_center"]] * n_comp) * units.Hz,
        spectral_index=[-0.8] * n_comp,
        stokes=np.repeat(np.reshape([1, 0, 0, 0], (4, 1, 1)), n_comp, axis=2)
        * units.Jy,
        frame="fk5",
        extra_column_dict={"image_x": image_x, "image_y": image_y},
    )
    new_sky = stokes_cnv(sky=sky, antenna=antenna, obs=obs, inverse=True)

    # I expected a slightly better tolerance than this...
    for pol_i in range(2):
        np.testing.assert_allclose(
            new_sky.extra_columns[f"flux_pol_{pol_i}"],
            np.full((n_comp,), 0.5, dtype=np.complex128),
            atol=0.02,
            rtol=0,
        )


@pytest.mark.github_actions
def test_source_dft_center():
    # check that a centered source returns all ones in uv plane

    model_uv = source_dft(
        x_loc=np.atleast_1d([2048.0 / 2]),
        y_loc=np.atleast_1d([2048.0 / 2]),
        dimension=2048,
        elements=2048,
        flux=np.reshape(np.array([0.5, 0.5, 0, 0]), (4, 1)),
    )

    np.testing.assert_allclose(model_uv[0:2], 0.5, atol=1e-15, rtol=0)
    np.testing.assert_allclose(model_uv[2:], 0, atol=1e-15, rtol=0)


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_source_dft_model(model_uv_zenith_2013):
    model_uv_full = model_uv_zenith_2013

    fhd_model_uv_file = fetch_data("2013_zenith_gleam_model_uv_small")
    fhd_model_uv_sav_dict = get_savs(fhd_model_uv_file, "")
    fhd_model_uv_sav_dict = recarray_to_dict(fhd_model_uv_sav_dict)
    fhd_model_uv_cut = fhd_model_uv_sav_dict["model_uv_cut"].T
    xrange = fhd_model_uv_sav_dict["xrange"]
    yrange = fhd_model_uv_sav_dict["yrange"]

    np.testing.assert_allclose(
        model_uv_full[xrange[0] : xrange[1] + 1, yrange[0] : yrange[1] + 1],
        fhd_model_uv_cut,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.github_actions
def test_vis_delay_filter(zenith_obs_2013, zenith_params_2013):
    obs = zenith_obs_2013
    params = zenith_params_2013

    cut_vis_model_file = fetch_data("2013_zenith_gleam_model_vis_75bl")
    cut_vis_model_dict = get_savs(cut_vis_model_file, "")
    cut_vis_model_dict = recarray_to_dict(cut_vis_model_dict)
    cut_vis_model = cut_vis_model_dict["vis_model_ptr"].transpose(0, 2, 1)

    nblt = cut_vis_model.shape[2]
    # make params lengths match cut down vis length
    for key in params:
        params[key] = params[key][:nblt]

    filtered_vis_arr = vis_delay_filter(cut_vis_model, params=params, obs=obs)

    fhd_filtered_vis_model_file = fetch_data(
        "2013_zenith_gleam_model_vis_75bl_filtered"
    )
    fhd_filtered_vis_model_dict = get_savs(fhd_filtered_vis_model_file, "")
    fhd_filtered_vis_model_dict = recarray_to_dict(fhd_filtered_vis_model_dict)
    fhd_filtered_vis_model = fhd_filtered_vis_model_dict["vis_model_ptr"].transpose(
        0, 2, 1
    )

    np.testing.assert_allclose(
        filtered_vis_arr, fhd_filtered_vis_model, atol=1e-10, rtol=0
    )


@pytest.mark.github_actions
def test_setup_uv_vals():
    img_size = 2048

    xvals, yvals = _setup_uv_vals(dimension=img_size, elements=img_size)

    assert xvals.shape == (img_size**2,)
    assert yvals.shape == (img_size**2,)
    assert xvals.min() == -1 * img_size / 2.0
    assert xvals.max() == img_size / 2.0 - 1
    assert yvals.min() == -1 * img_size / 2.0
    assert yvals.max() == img_size / 2.0 - 1

    psf_dim = 14
    uv_mask = np.full((img_size, img_size), True, dtype=bool)

    # use typical mask
    uv_mask = np.full((img_size, img_size), True)
    uv_mask[:, img_size // 2 + psf_dim :] = False
    uv_i_use = np.nonzero(uv_mask)

    xvals, yvals = _setup_uv_vals(
        dimension=img_size, elements=img_size, uv_i_use=uv_i_use
    )

    assert xvals.shape == (img_size * (img_size // 2 + psf_dim),)
    assert yvals.shape == (img_size * (img_size // 2 + psf_dim),)
    assert xvals.min() == -1 * img_size / 2.0
    assert xvals.max() == img_size / 2.0 - 1
    assert yvals.min() == -1 * img_size / 2.0
    assert yvals.max() == psf_dim - 1

    with pytest.raises(
        ValueError, match="either pass uv_i_use or xvals and yvals not both."
    ):
        xvals, yvals = _setup_uv_vals(
            dimension=img_size, elements=img_size, uv_i_use=uv_i_use, xvals=xvals
        )

    with pytest.raises(
        ValueError, match="If xvals or yvals is provided they must both be provided"
    ):
        xvals, yvals = _setup_uv_vals(
            dimension=img_size, elements=img_size, xvals=xvals
        )

    with pytest.raises(ValueError, match="xvals and yvals must have the same shape"):
        xvals, yvals = _setup_uv_vals(
            dimension=img_size, elements=img_size, xvals=xvals, yvals=yvals[0]
        )


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_vis_source_model_smoke(
    mwa_aee_beam_zenith_2013, zenith_params_2013, gleam_cut_2013_sky
):
    # This just checks that calling vis_source_model doesn't error
    antenna, psf, obs, pyfhd_config = mwa_aee_beam_zenith_2013
    sky = gleam_cut_2013_sky
    params = zenith_params_2013

    vis_source_model(
        pyfhd_config=pyfhd_config,
        obs=obs,
        psf=psf,
        params=params,
        antenna=antenna,
        skymodel=sky,
        vis_weights=None,
        fill_model_visibilities=True,
    )

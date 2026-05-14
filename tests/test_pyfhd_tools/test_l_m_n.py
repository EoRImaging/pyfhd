from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from pyfhd.pyfhd_tools.test_utils import get_data, get_data_items
from pyfhd.pyfhd_tools.pyfhd_utils import l_m_n


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "pyfhd_tools", "l_m_n")


def test_l_m_n_one(data_dir):
    obs, psf = get_data(
        data_dir, "visibility_grid_input_obs.npy", "visibility_grid_input_psf.npy"
    )
    expected_l_mode, expected_m_mode, expected_n_tracked = get_data_items(
        data_dir,
        "visibility_grid_output_l_mode.npy",
        "visibility_grid_output_m_mode.npy",
        "visibility_grid_output_n_tracked.npy",
    )
    obs = recarray_to_dict(obs)
    psf = recarray_to_dict(psf)
    l_mode, m_mode, n_tracked = l_m_n(obs, psf)
    # Set the threshold for single precision accuracy. Including rounding errors.
    npt.assert_allclose(l_mode, expected_l_mode, atol=1e-7)
    npt.assert_allclose(m_mode, expected_m_mode, atol=1e-7)
    npt.assert_allclose(n_tracked, expected_n_tracked, atol=1e-7)


def test_l_m_n_two(data_dir):
    obs, psf = get_data(data_dir, "input_obs_2.npy", "input_psf_2.npy")
    (
        obsdec,
        obsra,
        dec_arr,
        ra_arr,
        expected_l_mode,
        expected_m_mode,
        expected_n_tracked,
    ) = get_data_items(
        data_dir,
        "input_obsdec_2.npy",
        "input_obsra_2.npy",
        "input_dec_arr_2.npy",
        "input_ra_arr_2.npy",
        "output_l_mode_2.npy",
        "output_m_mode_2.npy",
        "output_n_tracked_2.npy",
    )
    obs = recarray_to_dict(obs)
    psf = recarray_to_dict(psf)
    l_mode, m_mode, n_tracked = l_m_n(
        obs,
        psf,
        obsdec=obsdec,
        obsra=obsra,
        declination_arr=dec_arr,
        right_ascension_arr=ra_arr,
    )
    # Set the threshold for single precision accuracy. Including rounding errors.
    npt.assert_allclose(l_mode, expected_l_mode, atol=2e-7)
    npt.assert_allclose(m_mode, expected_m_mode, atol=1e-7)
    npt.assert_allclose(n_tracked, expected_n_tracked, atol=1e-7)


def test_l_m_n_three(data_dir):
    obs, psf = get_data(data_dir, "input_obs_3.npy", "input_psf_3.npy")
    expected_l_mode, expected_m_mode, expected_n_tracked = get_data_items(
        data_dir, "output_l_mode_3.npy", "output_m_mode_3.npy", "output_n_tracked_3.npy"
    )
    obs = recarray_to_dict(obs)
    psf = recarray_to_dict(psf)
    l_mode, m_mode, n_tracked = l_m_n(obs, psf)
    # Set the threshold for single precision accuracy. Including rounding errors.
    npt.assert_allclose(l_mode, expected_l_mode, atol=2e-7)
    npt.assert_allclose(m_mode, expected_m_mode, atol=1e-7)
    npt.assert_allclose(n_tracked, expected_n_tracked, atol=1e-7)

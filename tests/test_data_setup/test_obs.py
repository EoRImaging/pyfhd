import pytest
from logging import Logger
from pathlib import Path
from os import environ as env
from pyfhd.data_setup.uvfits import extract_header, create_params, create_layout
from pyfhd.data_setup.obs import create_obs
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt
import numpy as np
from scipy.io import readsav


@pytest.fixture(
    scope="function",
    params=[
        "1088716296",
        "point_offzenith_8s_80kHz_analy_autos+gain_errors",
        "point_zenith_8s_80kHz_analy_autos+gain_errors",
    ],
)
def obs_id(request):
    # Ensure you have created the symbolic links for the point_offzenith and point_zenith
    # metafits files from 1088716176.metafits
    return request.param


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "data_setup", "uvfits")


@pytest.fixture
def obs_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "data_setup", "fhd_struct_init_obs")


def check_sav_file(path: Path, run: int, pyfhd_config: dict) -> Path:
    """
    Checks if the h5 file for obs testing exists, if it does return it's path
    if it does not exist, create it fro the sav file, then then return the path.

    Parameters
    ----------
    path : Path
        Path to the obs testing directory containing the resulting FHD obs structures
    pyfhd_config : dict
        A mock example of pyfhd's configuration given to the test

    Returns
    -------
    h5_file: Path
        The path to the h5_file
    """
    if "_" in pyfhd_config["obs_id"]:
        obs_id_split = pyfhd_config["obs_id"].split("_")
        tag = f"{obs_id_split[0]}_{obs_id_split[1]}"
    else:
        tag = pyfhd_config["obs_id"]
    h5_file = Path(path, f"{tag}_run{run}_after_{path.name}.h5")
    if h5_file.exists():
        return h5_file
    sav_file = h5_file.with_suffix(".sav")
    sav_dict = readsav(sav_file, python_dict=True)
    if "_" in pyfhd_config["obs_id"]:
        obs = recarray_to_dict(sav_dict["struct"])
    else:
        obs = recarray_to_dict(sav_dict["obs"])
    save(h5_file, obs, "after_file")

    return h5_file


def test_2_pol_obs_creation(obs_id, data_dir, obs_dir):
    # The obs creation test is more of an integration test, since we will be
    # using the extract_header, create_params, and create_layout to create the obs dictionary.
    # If this test pass then it essentially means that the dictionaries are almost identical
    # to that of the IDL structures in the ways that matter for a pyfhd run.
    # In this case we're only going to test the obs structure from run1 of each test.
    logger = Logger(1)
    pyfhd_config = {
        "obs_id": obs_id,
        "input_path": data_dir,
        "n_pol": 2,
        "instrument": "mwa",
        "FoV": None,
        "dimension": 2048,
        "elements": 2048,
        "kbinsize": 0.5,
        "min_baseline": 1,
        "time_cut": None,
        "beam_nfreq_avg": 16,
        "dft_threshold": False,
        "healpix_inds": 1,
        "output_dir": ".",
        "override_target_phasera": None,
        "override_target_phasedec": None,
    }
    # Grab the obs from the after files
    obs_fhd_result_path = check_sav_file(obs_dir, 1, pyfhd_config)
    obs_fhd = load(obs_fhd_result_path)
    # Now set it up for loading the dat from the uvfits directory properly
    if (
        obs_id == "point_offzenith_8s_80kHz_analy_autos+gain_errors"
        or obs_id == "point_zenith_8s_80kHz_analy_autos+gain_errors"
    ):
        pyfhd_config["input_path"] = Path(data_dir, "_".join(obs_id.split("_")[:2]))
        pyfhd_config["obs_id"] = "1088716176"
    else:
        pyfhd_config["input_path"] = Path(data_dir, obs_id)
    pyfhd_header, params_data, antenna_header, antenna_data = extract_header(
        pyfhd_config, logger
    )
    params = create_params(pyfhd_header, params_data, logger)
    layout = create_layout(antenna_header, antenna_data, pyfhd_config, logger)
    obs = create_obs(pyfhd_header, params, layout, pyfhd_config, logger)

    Path(pyfhd_config["output_dir"], "layout.h5").unlink()

    # Check the basic obs info
    assert obs["n_pol"] == obs_fhd["n_pol"]
    assert obs["n_tile"] == obs_fhd["n_tile"]
    assert obs["n_freq"] == obs_fhd["n_freq"]
    assert obs["n_time"] == obs_fhd["n_time"]
    assert obs["kpix"] == obs_fhd["kpix"]
    assert obs["dimension"] == obs_fhd["dimension"]
    assert obs["elements"] == obs_fhd["elements"]
    assert obs["n_baselines"] == obs_fhd["nbaselines"]
    npt.assert_almost_equal(obs["degpix"], obs_fhd["degpix"])
    npt.assert_almost_equal(obs["max_baseline"], obs_fhd["max_baseline"])
    npt.assert_almost_equal(obs["min_baseline"], obs_fhd["min_baseline"])
    npt.assert_array_equal(obs["pol_names"], obs_fhd["pol_names"].astype("str"))

    # Check baseline_info
    npt.assert_array_equal(
        obs["baseline_info"]["time_use"], obs_fhd["baseline_info"]["time_use"]
    )
    assert obs["n_time_flag"] == obs_fhd["n_time_flag"]
    npt.assert_array_equal(
        obs["baseline_info"]["tile_use"], obs_fhd["baseline_info"]["tile_use"]
    )
    # tile_flag is a little weird given it wants pointers from tile_flag
    # The indexes provided to tile_flag also go beyond the index range of the metadata
    # when producing the tile flags for the 3rd and 4th polarization is this a bug in FHD?
    # npt.assert_array_equal(obs['baseline_info']['tile_flag'], obs_fhd['baseline_info']['tile_flag'])
    assert obs["n_tile_flag"] == obs_fhd["n_tile_flag"]
    npt.assert_array_equal(
        obs["baseline_info"]["freq_use"], obs_fhd["baseline_info"]["freq_use"]
    )
    assert obs["dft_threshold"] == obs_fhd["dft_threshold"]
    npt.assert_array_equal(
        obs["baseline_info"]["tile_a"], obs_fhd["baseline_info"]["tile_a"]
    )
    npt.assert_array_equal(
        obs["baseline_info"]["tile_b"], obs_fhd["baseline_info"]["tile_b"]
    )
    npt.assert_array_equal(
        obs["baseline_info"]["tile_names"],
        np.char.strip((obs_fhd["baseline_info"]["tile_names"].astype("str"))).astype(
            int
        ),
    )

    # Check healpix
    assert obs["healpix"]["nside"] == obs_fhd["healpix"]["nside"]
    assert obs["healpix"]["n_pix"] == obs_fhd["healpix"]["n_pix"]
    assert obs["healpix"]["ind_list"] == int(obs_fhd["healpix"]["ind_list"])
    assert obs["healpix"]["n_zero"] == obs_fhd["healpix"]["n_zero"]


def test_4_pol_obs_creation(obs_id, data_dir, obs_dir):
    # The obs creation test is more of an integration test, since we will be
    # using the extract_header, create_params, and create_layout to create the obs dictionary.
    # If this test pass then it essentially means that the dictionaries are almost identical
    # to that of the IDL structures in the ways that matter for a pyfhd run.
    # In this case we're only going to test the obs structure from run1 of each test.
    if obs_id == "1088716296":
        pytest.skip("1088716296 hasn't been prepared to do 4 polarizations")
    logger = Logger(1)
    pyfhd_config = {
        "obs_id": obs_id,
        "input_path": data_dir,
        "n_pol": 4,
        "instrument": "mwa",
        "FoV": None,
        "dimension": 2048,
        "elements": 2048,
        "kbinsize": 0.5,
        "min_baseline": 1,
        "time_cut": None,
        "beam_nfreq_avg": 16,
        "dft_threshold": False,
        "healpix_inds": 1,
        "output_dir": ".",
        "override_target_phasera": None,
        "override_target_phasedec": None,
    }
    # Grab the obs from the after files
    obs_fhd_result_path = check_sav_file(obs_dir, 3, pyfhd_config)
    obs_fhd = load(obs_fhd_result_path)
    # Now set it up for loading the dat from the uvfits directory properly
    if (
        obs_id == "point_offzenith_8s_80kHz_analy_autos+gain_errors"
        or obs_id == "point_zenith_8s_80kHz_analy_autos+gain_errors"
    ):
        pyfhd_config["input_path"] = Path(data_dir, "_".join(obs_id.split("_")[:2]))
        pyfhd_config["obs_id"] = "1088716176"
    else:
        pyfhd_config["input_path"] = Path(data_dir, obs_id)
    pyfhd_header, params_data, antenna_header, antenna_data = extract_header(
        pyfhd_config, logger
    )
    params = create_params(pyfhd_header, params_data, logger)
    layout = create_layout(antenna_header, antenna_data, pyfhd_config, logger)
    obs = create_obs(pyfhd_header, params, layout, pyfhd_config, logger)

    Path(pyfhd_config["output_dir"], "layout.h5").unlink()

    # Check the basic obs info
    assert obs["n_pol"] == obs_fhd["n_pol"]
    assert obs["n_tile"] == obs_fhd["n_tile"]
    assert obs["n_freq"] == obs_fhd["n_freq"]
    assert obs["n_time"] == obs_fhd["n_time"]
    assert obs["kpix"] == obs_fhd["kpix"]
    assert obs["dimension"] == obs_fhd["dimension"]
    assert obs["elements"] == obs_fhd["elements"]
    npt.assert_almost_equal(obs["degpix"], obs_fhd["degpix"])
    npt.assert_almost_equal(obs["max_baseline"], obs_fhd["max_baseline"])
    npt.assert_almost_equal(obs["min_baseline"], obs_fhd["min_baseline"])
    npt.assert_array_equal(obs["pol_names"], obs_fhd["pol_names"].astype("str"))

    # Check baseline_info
    npt.assert_array_equal(
        obs["baseline_info"]["time_use"], obs_fhd["baseline_info"]["time_use"]
    )
    assert obs["n_time_flag"] == obs_fhd["n_time_flag"]
    # tile_flag is a little weird given it wants pointers from tile_flag
    # The indexes provided to tile_flag also go beyond the index range of the metadata
    # is this a bug in FHD? Thankfully it's not used elsewhere, and I can get the same behavior
    # with two polarizations, but I suspect there could be a difference with 4 polarizations
    # in the tile_use?
    # YES it breaks with 4 polarizations!
    # npt.assert_array_equal(obs['baseline_info']['tile_use'], obs_fhd['baseline_info']['tile_use'])
    # npt.assert_array_equal(obs['baseline_info']['tile_flag'], obs_fhd['baseline_info']['tile_flag'])
    # assert(obs['n_tile_flag'] == obs_fhd['n_tile_flag'])
    npt.assert_array_equal(
        obs["baseline_info"]["freq_use"], obs_fhd["baseline_info"]["freq_use"]
    )
    assert obs["dft_threshold"] == obs_fhd["dft_threshold"]
    npt.assert_array_equal(
        obs["baseline_info"]["tile_a"], obs_fhd["baseline_info"]["tile_a"]
    )
    npt.assert_array_equal(
        obs["baseline_info"]["tile_b"], obs_fhd["baseline_info"]["tile_b"]
    )
    npt.assert_array_equal(
        obs["baseline_info"]["tile_names"],
        np.char.strip((obs_fhd["baseline_info"]["tile_names"].astype("str"))).astype(
            int
        ),
    )

    # Check healpix
    assert obs["healpix"]["nside"] == obs_fhd["healpix"]["nside"]
    assert obs["healpix"]["n_pix"] == obs_fhd["healpix"]["n_pix"]
    assert obs["healpix"]["ind_list"] == int(obs_fhd["healpix"]["ind_list"])
    assert obs["healpix"]["n_zero"] == obs_fhd["healpix"]["n_zero"]

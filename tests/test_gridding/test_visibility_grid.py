from PyFHD.io.pyfhd_io import recarray_to_dict, convert_sav_to_dict
import pytest
import numpy.testing as npt
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.pyfhd_tools.test_utils import get_savs
from PyFHD.io.pyfhd_io import save, load
from logging import Logger
from scipy.io import readsav
import importlib_resources


@pytest.fixture
def data_dir():
    if env.get("PYFHD_TEST_PATH"):
        return Path(env.get("PYFHD_TEST_PATH"), "gridding", "visibility_grid")
    else:
        return None


@pytest.fixture(
    scope="function",
    params=[
        1,
        pytest.param(2, marks=pytest.mark.github_actions),
        pytest.param(3, marks=pytest.mark.github_actions),
        pytest.param(4, marks=pytest.mark.github_actions),
        pytest.param(5, marks=pytest.mark.github_actions),
        pytest.param(6, marks=pytest.mark.github_actions),
        pytest.param(7, marks=pytest.mark.github_actions),
    ],
)
def number(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def before_gridding(data_dir: Path, number: int, request: pytest.FixtureRequest):
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("PyFHD.resources.test_data").joinpath(
            "gridding", "visibility_grid"
        )
    before_gridding = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if before_gridding.exists():
        return before_gridding

    h5_save_dict = get_savs(data_dir, f"input_{number}.sav")
    # Subset the beam_ptr so we only take the first index of the baselines
    # which contains the pointer for the rest of the baselines
    h5_save_dict["psf"]["beam_ptr"][0] = h5_save_dict["psf"]["beam_ptr"][0].T[:, :, 0]
    h5_save_dict = recarray_to_dict(h5_save_dict)
    h5_save_dict["uniform_flag"] = (
        True
        if ("uniform_filter" in h5_save_dict and h5_save_dict["uniform_filter"])
        else False
    )
    h5_save_dict["no_conjugate"] = (
        True
        if ("no_conjugate" in h5_save_dict and h5_save_dict["no_conjugate"])
        else False
    )
    h5_save_dict["obs"]["n_baselines"] = h5_save_dict["obs"]["nbaselines"]
    # Transpose the model if it exists
    if "model_ptr" in h5_save_dict and h5_save_dict["model_ptr"] is not None:
        h5_save_dict["model_ptr"] = h5_save_dict["model_ptr"].T
    h5_save_dict["pyfhd_config"] = {
        "interpolate_kernel": h5_save_dict["psf"]["interpolate_kernel"],
        "psf_dim": h5_save_dict["psf"]["dim"],
        "psf_resolution": h5_save_dict["psf"]["resolution"],
        "beam_mask_threshold": h5_save_dict["psf"]["beam_mask_threshold"],
        "beam_clip_floor": h5_save_dict["extra"]["beam_clip_floor"],
        "image_filter": h5_save_dict["extra"]["image_filter_fn"],
        "mask_mirror_indices": False,
        "beam_per_baseline": (
            True
            if (
                "beam_per_baseline" in h5_save_dict
                and h5_save_dict["beam_per_baseline"]
            )
            else False
        ),
        "grid_spectral": (
            True
            if ("grid_spectral" in h5_save_dict and h5_save_dict["grid_spectral"])
            else False
        ),
        "grid_weights": True if h5_save_dict["weights"] else False,
        "grid_variance": (
            True if ("variance" in h5_save_dict and h5_save_dict["variance"]) else False
        ),
        "grid_uniform": (
            True
            if ("grid_uniform" in h5_save_dict and h5_save_dict["grid_uniform"])
            else False
        ),
    }
    h5_save_dict["visibility_ptr"] = h5_save_dict["visibility_ptr"].T
    h5_save_dict["vis_weight_ptr"] = h5_save_dict["vis_weight_ptr"].T
    if "fi_use" not in h5_save_dict:
        h5_save_dict["fi_use"] = None
    else:
        if not isinstance(h5_save_dict["fi_use"], np.ndarray):
            # Assume fi_use is an integer, make it an array
            h5_save_dict["fi_use"] = np.array([h5_save_dict["fi_use"]], dtype=np.int64)
    if "bi_use" not in h5_save_dict:
        h5_save_dict["bi_use"] = None
    else:
        if not isinstance(h5_save_dict["bi_use"], np.ndarray):
            h5_save_dict["bi_use"] = np.array([h5_save_dict["bi_use"]], dtype=np.int64)
    save(before_gridding, h5_save_dict, "before_file")

    return before_gridding


@pytest.fixture
def after_gridding(data_dir: Path, number: int, request: pytest.FixtureRequest):
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("PyFHD.resources.test_data").joinpath(
            "gridding", "visibility_grid"
        )
    after_gridding = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if after_gridding.exists():
        return after_gridding

    outputs = get_savs(data_dir, f"output_{number}.sav")
    # Delete the psf, we don't need it
    del outputs["psf"]
    del outputs["beam_arr"]
    del outputs["extra"]
    outputs = recarray_to_dict(outputs)

    h5_save_dict = {
        "image_uv": outputs["image_uv"],
        "weights": outputs["weights"],
        "variance": outputs["variance"],
        "uniform_filter": outputs["uniform_filter"],
        "nf_vis": outputs["obs"]["nf_vis"],
    }

    if "model_return" in outputs:
        h5_save_dict["model_return"] = outputs["model_return"]

    save(after_gridding, h5_save_dict, "after_file")

    return after_gridding


def test_visibility_grid(
    before_gridding: Path, after_gridding: Path, request: pytest.FixtureRequest
):
    # This was done here to make it work in GitHub Actions
    if request.node.get_closest_marker("github_actions"):
        data_dir = importlib_resources.files("PyFHD.resources.test_data").joinpath(
            "gridding", "visibility_grid"
        )
        before_gridding = Path(data_dir, before_gridding.name)
        after_gridding = Path(data_dir, after_gridding.name)
    h5_before = load(before_gridding)
    h5_after = load(after_gridding)

    h5_before["psf"]["id"] = h5_before["psf"]["id"].T

    # Format the indexing arrays if needed
    if h5_before["fi_use"] is not None and h5_before["fi_use"].size == 1:
        new_arr = np.zeros(1, dtype=np.int64)
        new_arr[0] = h5_before["fi_use"][0]
        h5_before["fi_use"] = new_arr

    if h5_before["bi_use"] is not None and h5_before["bi_use"].size == 1:
        new_arr = np.zeros(1, dtype=np.int64)
        new_arr[0] = h5_before["bi_use"][0]
        h5_before["bi_use"] = new_arr

    obs = recarray_to_dict(h5_before["obs"])
    psf = recarray_to_dict(h5_before["psf"])

    gridding_dict = visibility_grid(
        h5_before["visibility_ptr"],
        h5_before["vis_weight_ptr"],
        obs,
        psf,
        h5_before["params"],
        h5_before["polarization"],
        h5_before["pyfhd_config"],
        Logger(1),
        uniform_flag=h5_before["uniform_flag"],
        no_conjugate=h5_before["no_conjugate"],
        model=h5_before["model_ptr"],
        fi_use=h5_before["fi_use"],
        bi_use=h5_before["bi_use"],
    )
    # All atols are done by the lowest precision that passed for ALL tests
    npt.assert_allclose(gridding_dict["image_uv"], h5_after["image_uv"], atol=1.5e-7)
    npt.assert_allclose(gridding_dict["weights"], h5_after["weights"], atol=1e-8)
    npt.assert_allclose(gridding_dict["variance"], h5_after["variance"], atol=1e-8)
    # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
    # The minor difference in bin_n affected the uniform filter. The precision difference could cause errors upto 1
    # This doesn't occur for every test.
    npt.assert_allclose(gridding_dict["obs"]["nf_vis"], h5_after["nf_vis"], atol=1e-8)
    npt.assert_allclose(
        gridding_dict["uniform_filter"], h5_after["uniform_filter"], atol=0.5
    )

    if "model_return" in gridding_dict:
        npt.assert_allclose(
            gridding_dict["model_return"], h5_after["model_return"], atol=1e-7
        )


# FULL SIZE TESTS BELOW


@pytest.fixture(scope="function", params=[1])
def full_number(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def full_before_gridding(data_dir: Path, full_number: int):
    before_gridding = Path(
        data_dir, f"test_full_size_{full_number}_before_{data_dir.name}.h5"
    )

    if before_gridding.exists():
        return before_gridding

    h5_save_dict = get_savs(data_dir, f"full_size_input_{full_number}.sav")
    # Subset the beam_ptr so we only take the first index of the baselines
    # which contains the pointer for the rest of the baselines
    h5_save_dict["psf"]["beam_ptr"][0] = h5_save_dict["psf"]["beam_ptr"][0].T[:, :, 0]
    h5_save_dict = recarray_to_dict(h5_save_dict)
    h5_save_dict["uniform_flag"] = (
        True
        if ("uniform_filter" in h5_save_dict and h5_save_dict["uniform_filter"])
        else False
    )
    h5_save_dict["no_conjugate"] = (
        True
        if ("no_conjugate" in h5_save_dict and h5_save_dict["no_conjugate"])
        else False
    )
    h5_save_dict["obs"]["n_baselines"] = h5_save_dict["obs"]["nbaselines"]
    # Transpose the model if it exists
    if "model_ptr" in h5_save_dict and h5_save_dict["model_ptr"] is not None:
        h5_save_dict["model_ptr"] = h5_save_dict["model_ptr"].T
    else:
        h5_save_dict["model_ptr"] = None
    h5_save_dict["pyfhd_config"] = {
        "interpolate_kernel": h5_save_dict["psf"]["interpolate_kernel"],
        "psf_dim": h5_save_dict["psf"]["dim"],
        "psf_resolution": h5_save_dict["psf"]["resolution"],
        "beam_mask_threshold": h5_save_dict["psf"]["beam_mask_threshold"],
        "beam_clip_floor": h5_save_dict["extra"]["beam_clip_floor"],
        "image_filter": h5_save_dict["extra"]["image_filter_fn"],
        "mask_mirror_indices": False,
        "beam_per_baseline": (
            True
            if (
                "beam_per_baseline" in h5_save_dict
                and h5_save_dict["beam_per_baseline"]
            )
            else False
        ),
        "grid_spectral": (
            True
            if ("grid_spectral" in h5_save_dict and h5_save_dict["grid_spectral"])
            else False
        ),
        "grid_weights": True if h5_save_dict["weights"] else False,
        "grid_variance": (
            True if ("variance" in h5_save_dict and h5_save_dict["variance"]) else False
        ),
        "grid_uniform": (
            True
            if ("grid_uniform" in h5_save_dict and h5_save_dict["grid_uniform"])
            else False
        ),
    }
    h5_save_dict["visibility_ptr"] = h5_save_dict["visibility_ptr"].T
    h5_save_dict["vis_weight_ptr"] = h5_save_dict["vis_weight_ptr"].T
    if "fi_use" not in h5_save_dict:
        h5_save_dict["fi_use"] = None
    else:
        if not isinstance(h5_save_dict["fi_use"], np.ndarray):
            # Assume fi_use is an integer, make it an array
            h5_save_dict["fi_use"] = np.array([h5_save_dict["fi_use"]], dtype=np.int64)
    if "bi_use" not in h5_save_dict:
        h5_save_dict["bi_use"] = None
    else:
        if not isinstance(h5_save_dict["bi_use"], np.ndarray):
            h5_save_dict["bi_use"] = np.array([h5_save_dict["bi_use"]], dtype=np.int64)
    save(before_gridding, h5_save_dict, "before_file")

    return before_gridding


@pytest.fixture
def full_after_gridding(data_dir: Path, full_number: int):
    after_gridding = Path(
        data_dir, f"test_full_size_{full_number}_after_{data_dir.name}.h5"
    )

    if after_gridding.exists():
        return after_gridding

    outputs = recarray_to_dict(
        get_savs(data_dir, f"full_size_output_{full_number}.sav")
    )

    h5_save_dict = {
        "image_uv": outputs["image_uv"],
        "weights": outputs["weights"],
        "variance": outputs["variance"],
        # 'uniform_filter': outputs['uniform_filter'],
        "nf_vis": outputs["obs"]["nf_vis"],
    }

    if "model_return" in outputs:
        h5_save_dict["model_return"] = outputs["model_return"]

    save(after_gridding, h5_save_dict, "after_file")

    return after_gridding


def test_full_visibility_grid(full_before_gridding: Path, full_after_gridding: Path):
    h5_before = load(full_before_gridding)
    h5_after = load(full_after_gridding)

    # h5_before["psf"]["id"] = h5_before["psf"]["id"].T

    # Format the indexing arrays if needed
    if h5_before["fi_use"] is not None and h5_before["fi_use"].size == 1:
        new_arr = np.zeros(1, dtype=np.int64)
        new_arr[0] = h5_before["fi_use"]
        h5_before["fi_use"] = new_arr

    if h5_before["bi_use"] is not None and h5_before["bi_use"].size == 1:
        new_arr = np.zeros(1, dtype=np.int64)
        new_arr[0] = h5_before["bi_use"]
        h5_before["bi_use"] = new_arr

    gridding_dict = visibility_grid(
        h5_before["visibility_ptr"],
        h5_before["vis_weight_ptr"],
        h5_before["obs"],
        h5_before["psf"],
        h5_before["params"],
        h5_before["polarization"],
        h5_before["pyfhd_config"],
        Logger(1),
        uniform_flag=h5_before["uniform_flag"],
        no_conjugate=h5_before["no_conjugate"],
        model=h5_before["model_ptr"],
        fi_use=h5_before["fi_use"],
        bi_use=h5_before["bi_use"],
    )
    # All atols are done by the lowest precision that passed for ALL tests
    npt.assert_allclose(gridding_dict["image_uv"], h5_after["image_uv"], atol=1e-8)
    npt.assert_allclose(gridding_dict["weights"], h5_after["weights"], atol=1e-8)
    npt.assert_allclose(gridding_dict["variance"], h5_after["variance"], atol=1e-8)
    npt.assert_allclose(gridding_dict["obs"]["nf_vis"], h5_after["nf_vis"], atol=1e-8)
    # npt.assert_allclose(gridding_dict['uniform_filter'], h5_after['uniform_filter'], atol = 1e-8)

    if "model_return" in gridding_dict:
        npt.assert_allclose(
            gridding_dict["model_return"], h5_after["model_return"], atol=1e-8
        )


# VIS_MODEL_FREQ_SPLIT VISIBILITY GRID TESTS BELOW


@pytest.fixture(
    scope="function",
    params=["point_zenith", "point_offzenith", "1088285600", "1088716296"],
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3", "run4"])
def run(request):
    return request.param


vis_model_skip_tests: list[list[str]] = [
    ["point_zenith", "run3"],
    ["point_offzenith", "run3"],
    ["1088285600", "run4"],
    ["1088716296", "run3"],
]


@pytest.fixture()
def before_vis_model_freq_gridding(tag, run, data_dir):
    if [tag, run] in vis_model_skip_tests:
        return None
    before_file = Path(
        data_dir, f"{tag}_{run}_before_{data_dir.name}_vis_model_freq_split.h5"
    )
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")

    h5_save_dict = readsav(sav_file, python_dict=True)
    h5_save_dict = recarray_to_dict(h5_save_dict)

    h5_save_dict["uniform_flag"] = (
        True
        if ("uniform_filter" in h5_save_dict and h5_save_dict["uniform_filter"])
        else False
    )
    h5_save_dict["no_conjugate"] = (
        True
        if ("no_conjugate" in h5_save_dict and h5_save_dict["no_conjugate"])
        else False
    )
    h5_save_dict["obs_out"]["n_baselines"] = h5_save_dict["obs_out"]["nbaselines"]
    h5_save_dict["obs"] = h5_save_dict["obs_out"]
    del h5_save_dict["obs_out"]
    # Transpose the model if it exists
    if "model_ptr" in h5_save_dict and h5_save_dict["model_ptr"] is not None:
        h5_save_dict["model_ptr"] = h5_save_dict["model_ptr"].T
    psf = load(
        Path(env.get("PYFHD_TEST_PATH"), "beams", "decomp_beam_pointing0.h5"),
        None,
        lazy_load=True,
    )
    h5_save_dict["pyfhd_config"] = {
        "interpolate_kernel": psf["interpolate_kernel"][0],
        "psf_dim": psf["dim"][0],
        "psf_resolution": psf["resolution"][0],
        "beam_mask_threshold": psf["beam_mask_threshold"][0],
        "beam_clip_floor": h5_save_dict["extra"]["beam_clip_floor"],
        "image_filter": h5_save_dict["extra"]["image_filter_fn"],
        "mask_mirror_indices": False,
        "beam_per_baseline": (
            True
            if (
                "beam_per_baseline" in h5_save_dict
                and h5_save_dict["beam_per_baseline"]
            )
            else False
        ),
        "grid_spectral": (
            True
            if ("grid_spectral" in h5_save_dict and h5_save_dict["grid_spectral"])
            else False
        ),
        "grid_weights": True if h5_save_dict["weights_holo"] else False,
        "grid_variance": (
            True
            if ("variance_holo" in h5_save_dict and h5_save_dict["variance_holo"])
            else False
        ),
        "grid_uniform": (
            True
            if ("grid_uniform" in h5_save_dict and h5_save_dict["grid_uniform"])
            else False
        ),
    }
    h5_save_dict["visibility_ptr"] = h5_save_dict["vis_ptr"].T
    h5_save_dict["vis_weight_ptr"] = np.swapaxes(
        h5_save_dict["vis_weights_use"], -1, -2
    )[0]
    del h5_save_dict["vis_ptr"]
    del h5_save_dict["vis_weights_use"]
    if "fi_use" not in h5_save_dict:
        h5_save_dict["fi_use"] = None
    else:
        if not isinstance(h5_save_dict["fi_use"], np.ndarray):
            # Assume fi_use is an integer, make it an array
            h5_save_dict["fi_use"] = np.array([h5_save_dict["fi_use"]], dtype=np.int64)
    if "bi_use" not in h5_save_dict:
        h5_save_dict["bi_use"] = None
    else:
        if not isinstance(h5_save_dict["bi_use"], np.ndarray):
            h5_save_dict["bi_use"] = np.array([h5_save_dict["bi_use"]], dtype=np.int64)

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture()
def after_vis_model_freq_gridding(tag, run, data_dir):
    if [tag, run] in vis_model_skip_tests:
        return None
    after_file = Path(
        data_dir, f"{tag}_{run}_after_{data_dir.name}_vis_model_freq_split.h5"
    )
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    save(after_file, sav_dict, "after_file")

    return after_file


def test_visibility_grid_in_vis_model_freq_split(
    before_vis_model_freq_gridding, after_vis_model_freq_gridding
):
    if before_vis_model_freq_gridding == None or after_vis_model_freq_gridding == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {vis_model_skip_tests}"
        )
    h5_before = load(before_vis_model_freq_gridding)
    h5_after = load(after_vis_model_freq_gridding)

    psf = load(
        Path(env.get("PYFHD_TEST_PATH"), "beams", "decomp_beam_pointing0.h5"),
        None,
        lazy_load=True,
    )

    # Format the indexing arrays if needed
    if h5_before["fi_use"] is not None and h5_before["fi_use"].size == 1:
        new_arr = np.zeros(1, dtype=np.int64)
        new_arr[0] = h5_before["fi_use"]
        h5_before["fi_use"] = new_arr

    if h5_before["bi_use"] is not None and h5_before["bi_use"].size == 1:
        new_arr = np.zeros(1, dtype=np.int64)
        new_arr[0] = h5_before["bi_use"]
        h5_before["bi_use"] = new_arr

    gridding_dict = visibility_grid(
        h5_before["visibility_ptr"],
        h5_before["vis_weight_ptr"],
        h5_before["obs"],
        psf,
        h5_before["params"],
        0,
        h5_before["pyfhd_config"],
        Logger(1),
        uniform_flag=h5_before["uniform_flag"],
        no_conjugate=h5_before["no_conjugate"],
        model=h5_before["model_ptr"],
        fi_use=h5_before["fi_use"],
        bi_use=h5_before["bi_use"],
    )
    # All atols are done by the lowest precision that passed for ALL tests
    npt.assert_allclose(gridding_dict["image_uv"], h5_after["dirty_uv"], atol=1.5e-7)
    npt.assert_allclose(gridding_dict["weights"], h5_after["weights_holo"], atol=1e-8)
    npt.assert_allclose(gridding_dict["variance"], h5_after["variance_holo"], atol=1e-8)
    npt.assert_allclose(
        gridding_dict["model_return"], h5_after["model_return"], atol=1e-8
    )
    # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
    # The minor difference in bin_n affected the uniform filter. The precision difference could cause errors upto 1
    # This doesn't occur for every test.
    npt.assert_allclose(gridding_dict["n_vis"], h5_after["n_vis"], atol=1e-8)

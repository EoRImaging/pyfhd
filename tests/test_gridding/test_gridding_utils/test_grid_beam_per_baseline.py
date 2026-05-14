from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.gridding.gridding_utils import grid_beam_per_baseline
from pyfhd.pyfhd_tools.test_utils import get_data, get_data_items
from pyfhd.io.pyfhd_io import save, load
from numpy.testing import assert_allclose
from logging import Logger


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "gridding", "grid_beam_per_baseline")


@pytest.fixture(scope="function", params=[1, 2, 3])
def number(request):
    return request.param


@pytest.fixture
def before_grid_per_baseline(data_dir, number):
    before_grid_per_baseline = Path(
        data_dir, f"test_{number}_before_{data_dir.name}.h5"
    )

    if before_grid_per_baseline.exists():
        return before_grid_per_baseline

    # For now put psf into dict
    psf, extra = get_data(
        data_dir,
        f"visibility_grid_psf_input_{number}.npy",
        f"visibility_grid_extra_input_{number}.npy",
    )
    psf = recarray_to_dict(psf)
    extra = recarray_to_dict(extra)
    pyfhd_config = {
        "psf_dim": psf["dim"],
        "psf_resolution": psf["resolution"],
        "beam_mask_threshold": psf["beam_mask_threshold"],
        "beam_clip_floor": True if extra["beam_clip_floor"] else False,
        "image_filter": "filter_uv_uniform",
    }
    (
        uu,
        vv,
        ww,
        l_mode,
        m_mode,
        n_tracked,
        frequency_array,
        x,
        y,
        xmin_use,
        ymin_use,
        freq_i,
        bt_index,
        polarization,
        fbin,
        image_bot,
        image_top,
        psf_dim3,
        box_matrix,
        vis_n,
    ) = get_data_items(
        data_dir,
        f"visibility_grid_uu_input_{number}.npy",
        f"visibility_grid_vv_input_{number}.npy",
        f"visibility_grid_ww_input_{number}.npy",
        f"visibility_grid_l_mode_input_{number}.npy",
        f"visibility_grid_m_mode_input_{number}.npy",
        f"visibility_grid_n_tracked_input_{number}.npy",
        f"visibility_grid_frequency_array_input_{number}.npy",
        f"visibility_grid_x_input_{number}.npy",
        f"visibility_grid_y_input_{number}.npy",
        f"visibility_grid_xmin_use_input_{number}.npy",
        f"visibility_grid_ymin_use_input_{number}.npy",
        f"visibility_grid_freq_i_input_{number}.npy",
        f"visibility_grid_bt_index_input_{number}.npy",
        f"visibility_grid_polarization_input_{number}.npy",
        f"visibility_grid_fbin_input_{number}.npy",
        f"visibility_grid_image_bot_input_{number}.npy",
        f"visibility_grid_image_top_input_{number}.npy",
        f"visibility_grid_psf_dim3_input_{number}.npy",
        f"visibility_grid_box_matrix_input_{number}.npy",
        f"visibility_grid_vis_n_input_{number}.npy",
    )

    h5_save_dict = {
        "psf": psf,
        "pyfhd_config": pyfhd_config,
        "uu": uu,
        "vv": vv,
        "ww": ww,
        "l_mode": l_mode,
        "m_mode": m_mode,
        "n_tracked": n_tracked,
        "frequency_array": frequency_array,
        "x": x,
        "y": y,
        "xmin_use": xmin_use,
        "ymin_use": ymin_use,
        "freq_i": freq_i,
        "bt_index": bt_index,
        "polarization": polarization,
        "fbin": fbin,
        "image_bot": int(image_bot),
        "image_top": int(image_top),
        "psf_dim3": psf_dim3,
        "box_matrix": box_matrix,
        "vis_n": vis_n,
    }

    save(before_grid_per_baseline, h5_save_dict, "before_file")

    return before_grid_per_baseline


@pytest.fixture
def after_grid_per_baseline(data_dir, number):
    after_grid_per_baseline = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if after_grid_per_baseline.exists():
        return after_grid_per_baseline

    h5_save_dict = {
        "box_matrix": get_data_items(
            data_dir, f"visibility_grid_box_matrix_output_{number}.npy"
        )
    }

    save(after_grid_per_baseline, h5_save_dict, "box_matrix")

    return after_grid_per_baseline


def test_grid_per_baseline(
    before_grid_per_baseline: Path, after_grid_per_baseline: Path
):
    h5_before = load(before_grid_per_baseline)
    expected_box_matrix = load(after_grid_per_baseline)

    output_box_matrix = grid_beam_per_baseline(
        h5_before["psf"],
        h5_before["pyfhd_config"],
        Logger(1),
        h5_before["uu"],
        h5_before["vv"],
        h5_before["ww"],
        h5_before["l_mode"],
        h5_before["m_mode"],
        h5_before["n_tracked"],
        h5_before["frequency_array"],
        h5_before["x"],
        h5_before["y"],
        h5_before["xmin_use"],
        h5_before["ymin_use"],
        h5_before["freq_i"],
        h5_before["bt_index"],
        h5_before["polarization"],
        h5_before["image_bot"],
        h5_before["image_top"],
        h5_before["psf_dim3"],
        h5_before["box_matrix"],
        h5_before["vis_n"],
    )

    assert_allclose(output_box_matrix, expected_box_matrix, atol=1e-8)

import pytest
from os import environ as env
from pathlib import Path
from pyfhd.gridding.gridding_utils import dirty_image_generate
from pyfhd.pyfhd_tools.test_utils import get_data_items
from numpy.testing import assert_allclose
from pyfhd.io.pyfhd_io import save, load
from logging import Logger


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "gridding", "dirty_image_generate")


@pytest.fixture(scope="function", params=[1, 2, 3, 4, 5, 6, 7, 8, 9])
def number(request):
    return request.param


def get_file(data_dir, file_name):
    if Path(data_dir, file_name).exists():
        item = get_data_items(data_dir, file_name)
        return item
    else:
        return None


@pytest.fixture
def dirty_before(data_dir, number):
    dirty_before = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if dirty_before.exists():
        return dirty_before

    pyfhd_config = {"image_filter": "filter_uv_uniform"}
    baseline_threshold = get_file(data_dir, f"input_baseline_threshold_{number}.npy")
    h5_save_dict = {
        "dirty_image_uv": get_file(data_dir, f"input_dirty_image_uv_{number}.npy"),
        "pyfhd_config": pyfhd_config,
        "mask": get_file(data_dir, f"input_mask_{number}.npy"),
        "baseline_threshold": baseline_threshold if baseline_threshold else 0,
        "normalization": get_file(data_dir, f"output_normalization_{number}.npy"),
        "resize": get_file(data_dir, f"input_resize_{number}.npy"),
        "width_smooth": get_file(data_dir, f"input_width_smooth_{number}.npy"),
        "degpix": get_file(data_dir, f"input_degpix_{number}.npy"),
        "not_real": (
            True if get_file(data_dir, f"input_no_real_{number}.npy") else False
        ),
        "pad_uv_image": get_file(data_dir, f"input_pad_uv_image_{number}.npy"),
        "weights": get_file(data_dir, f"input_weights_{number}.npy"),
        "filter": get_file(data_dir, f"input_filter_{number}.npy"),
        "beam_ptr": get_file(data_dir, f"input_beam_ptr_{number}.npy"),
    }

    save(dirty_before, h5_save_dict, "before_file")

    return dirty_before


@pytest.fixture
def dirty_after(data_dir, number):
    dirty_after = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if dirty_after.exists():
        return dirty_after

    dirty_image = get_data_items(data_dir, f"output_dirty_image_{number}.npy")
    normalization = get_file(data_dir, f"output_normalization_{number}.npy")
    h5_save_dict = {"dirty_image": dirty_image, "normalization": normalization}

    save(dirty_after, h5_save_dict, "after_file")

    return dirty_after


def test_dirty_image_generate(dirty_before: Path, dirty_after: Path):
    h5_before = load(dirty_before)
    h5_after = load(dirty_after)

    dirty_image, _, normalization = dirty_image_generate(
        h5_before["dirty_image_uv"],
        h5_before["pyfhd_config"],
        Logger(1),
        mask=h5_before["mask"],
        baseline_threshold=h5_before["baseline_threshold"],
        normalization=h5_before["normalization"],
        resize=int(h5_before["resize"]) if h5_before["resize"] else None,
        width_smooth=h5_before["width_smooth"],
        degpix=h5_before["degpix"],
        not_real=h5_before["not_real"],
        pad_uv_image=h5_before["pad_uv_image"],
        weights=h5_before["weights"],
        filter=h5_before["filter"],
        beam_ptr=h5_before["beam_ptr"],
    )

    assert_allclose(dirty_image, h5_after["dirty_image"], atol=1e-8)
    if h5_after["normalization"] is not None:
        assert_allclose(normalization, h5_after["normalization"], atol=1e-8)

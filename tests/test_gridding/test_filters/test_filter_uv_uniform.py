import pytest
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from pyfhd.gridding.filters import filter_uv_uniform
from pyfhd.pyfhd_tools.test_utils import get_data_items
from pyfhd.io.pyfhd_io import save, load


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "gridding", "filter_uv_uniform")


@pytest.fixture(scope="function", params=[2, 3])
def number(request):
    return request.param


def get_file(data_dir, file_name):
    if Path(data_dir, file_name).exists():
        item = get_data_items(data_dir, file_name)
        return item
    else:
        return None


@pytest.fixture
def filter_uni_before(data_dir, number):
    filter_uni_before = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if filter_uni_before.exists():
        return filter_uni_before

    image_uv, vis_count = get_data_items(
        data_dir,
        f"input_image_uv_{number}.npy",
        f"input_vis_count_{number}.npy",
    )

    weights = get_file(data_dir, f"input_weights_{number}.npy")

    h5_save_dict = {}
    h5_save_dict["image_uv"] = image_uv
    h5_save_dict["vis_count"] = vis_count
    h5_save_dict["weights"] = weights

    save(filter_uni_before, h5_save_dict, "before_file")

    return filter_uni_before


@pytest.fixture
def filter_uni_after(data_dir, number):
    filter_uni_after = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if filter_uni_after.exists():
        return filter_uni_after

    expected_image_uv, expected_filter = get_data_items(
        data_dir,
        f"output_image_uv_filtered_{number}.npy",
        f"output_filter_{number}.npy",
    )

    h5_save_dict = {}
    h5_save_dict["image_uv"] = expected_image_uv
    h5_save_dict["filter"] = expected_filter

    save(filter_uni_after, h5_save_dict, "after_file")

    return filter_uni_after


def test_filter_uv_uniform(filter_uni_before: Path, filter_uni_after: Path):
    h5_before = load(filter_uni_before)
    h5_after = load(filter_uni_after)

    image_uv_filtered, filter = filter_uv_uniform(
        h5_before["image_uv"], h5_before["vis_count"], weights=h5_before["weights"]
    )

    npt.assert_allclose(image_uv_filtered, h5_after["image_uv"], atol=2e-5)
    npt.assert_allclose(filter, h5_after["filter"], atol=1e-8)

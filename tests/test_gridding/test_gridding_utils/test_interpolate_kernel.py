from numpy.testing import assert_allclose
from pathlib import Path
import pytest
from pyfhd.gridding.gridding_utils import interpolate_kernel
from pyfhd.pyfhd_tools.test_utils import get_data_items
from pyfhd.io.pyfhd_io import save, load, recarray_to_dict
import importlib_resources


@pytest.fixture
def data_dir():
    return importlib_resources.files("pyfhd.resources.test_data").joinpath(
        "gridding", "interpolate_kernel"
    )


@pytest.fixture(scope="function", params=[1, 2, 3])
def number(request):
    return request.param


@pytest.fixture
def interp_kernel_before(data_dir, number):
    interp_kernel_before = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if interp_kernel_before.exists():
        return interp_kernel_before

    kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1 = get_data_items(
        data_dir,
        f"visibility_grid_input_kernel_arr_{number}.npy",
        f"visibility_grid_input_x_offset_{number}.npy",
        f"visibility_grid_input_y_offset_{number}.npy",
        f"visibility_grid_input_dx0dy0_{number}.npy",
        f"visibility_grid_input_dx1dy0_{number}.npy",
        f"visibility_grid_input_dx0dy1_{number}.npy",
        f"visibility_grid_input_dx1dy1_{number}.npy",
    )

    h5_save_dict = {
        "kernel_arr": kernel_arr,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "dx0dy0": dx0dy0,
        "dx1dy0": dx1dy0,
        "dx0dy1": dx0dy1,
        "dx1dy1": dx1dy1,
    }

    h5_save_dict = recarray_to_dict(h5_save_dict)

    save(interp_kernel_before, h5_save_dict, "before_file")

    return interp_kernel_before


@pytest.fixture
def interp_kernel_after(data_dir, number):
    interp_kernel_after = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if interp_kernel_after.exists():
        return interp_kernel_after

    h5_save_dict = {
        "kernel": get_data_items(
            data_dir, f"visibility_grid_output_kernel_{number}.npy"
        )
    }

    save(interp_kernel_after, h5_save_dict, "after_file")

    return interp_kernel_after


@pytest.mark.github_actions
def test_interpolate_kernel(interp_kernel_before: Path, interp_kernel_after: Path):
    h5_before = load(interp_kernel_before)
    expected_kernel = load(interp_kernel_after)

    kernel = interpolate_kernel(
        h5_before["kernel_arr"],
        h5_before["x_offset"],
        h5_before["y_offset"],
        h5_before["dx0dy0"],
        h5_before["dx1dy0"],
        h5_before["dx0dy1"],
        h5_before["dx1dy1"],
    )

    # Slightly higher absolute error than others due to precision differences
    # mostly caused by x_offset and y_offset
    assert_allclose(kernel, expected_kernel, atol=4e-4)

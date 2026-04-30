import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from pyfhd.gridding.gridding_utils import conjugate_mirror
from pyfhd.pyfhd_tools.test_utils import get_data_items
from pyfhd.io.pyfhd_io import save, load


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "gridding", "conjugate_mirror")


@pytest.fixture(scope="function", params=[1, 2, 3])
def number(request):
    return request.param


@pytest.fixture
def conjugate_file(data_dir, number):
    conjugate_file = Path(data_dir, f"test_{number}_{data_dir.name}.h5")

    if conjugate_file.exists():
        return conjugate_file

    input, conj_mirror_image = get_data_items(
        data_dir,
        f"visibility_grid_input_{number}.npy",
        f"visibility_grid_output_{number}.npy",
    )

    h5_save_dict = {"input": input, "conj_mirror_image": conj_mirror_image}

    save(conjugate_file, h5_save_dict, "file")

    return conjugate_file


def test_conjugate_mirror(conjugate_file: Path):
    h5 = load(conjugate_file)
    image = conjugate_mirror(h5["input"])
    assert np.array_equal(image, h5["conj_mirror_image"])

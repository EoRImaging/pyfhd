from pathlib import Path

import pytest

from pyfhd.io.pyfhd_io import (
    CHECKPOINTS,
    SAVEFILES,
    checkpoint_complete,
    checkpoint_filenames,
    product_file,
)


@pytest.mark.github_actions
@pytest.mark.parametrize("checkpoint", CHECKPOINTS.keys())
def test_checkpoint_files(tmp_path, checkpoint):
    pyfhd_config = {"output_dir": tmp_path, "obs_id": "1061316296"}
    file_dict = checkpoint_filenames(checkpoint, pyfhd_config, mkdir=True)

    for var, file in file_dict.items():
        if var == "healpix_cube":
            hpx_files = []
            for cube in ["hpx_even", "hpx_odd"]:
                for pol_name in ["XX", "YY"]:
                    hpx_files.append(f"{pyfhd_config['obs_id']}_{cube}_{pol_name}.h5")
            healpix_folder = Path(
                pyfhd_config["output_dir"], SAVEFILES["healpix_cube"]["folder"]
            )
            assert healpix_folder.exists()
            for fname in hpx_files:
                file = healpix_folder / fname
                file.touch()
        else:
            file.touch()

    assert checkpoint_complete(checkpoint, pyfhd_config)


@pytest.mark.github_actions
def test_other_files(tmp_path):
    pyfhd_config = {"output_dir": tmp_path, "obs_id": "1061316296"}

    var_names = list(SAVEFILES.keys())
    for var_list in CHECKPOINTS.values():
        for var in var_list:
            var_names.remove(var)

    for var in var_names:
        file = product_file(var, pyfhd_config, mkdir=True)
        file.touch()

        assert file.exists()

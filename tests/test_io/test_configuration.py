import importlib_resources
import sys
import time

import configargparse
import pytest

from pyfhd.pyfhd_tools.pyfhd_setup import git_info, pyfhd_parser, pyfhd_setup
from pyfhd.pyfhd import setup_directory


@pytest.mark.github_actions
@pytest.mark.parametrize(
    ("git_string", "output_dict"),
    [
        (
            "1.0.3.dev321+gc96fd9451.hmf",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "commit_str": "c96fd9451 (branch: hmf)",
                "branch": "hmf",
                "dirty_flag": False,
            },
        ),
        (
            "1.0.3.dev321+gc96fd9451.hmf.dirty",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "commit_str": "c96fd9451 (branch: hmf) DIRTY (uncommitted changes)",
                "branch": "hmf",
                "dirty_flag": True,
            },
        ),
        (
            "1.0.3.dev321+gc96fd9451",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "commit_str": "c96fd9451",
                "branch": None,
                "dirty_flag": False,
            },
        ),
        (
            "1.0.3.dev321+gc96fd9451.dirty",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "commit_str": "c96fd9451 DIRTY (uncommitted changes)",
                "branch": None,
                "dirty_flag": True,
            },
        ),
        (
            "1.0.2",
            {
                "tag": "1.0.2",
                "commit": None,
                "commit_str": "1.0.2",
                "branch": None,
                "dirty_flag": False,
            },
        ),
    ],
)
def test_git_info(git_string, output_dict):
    version_info = git_info(git_string)

    assert version_info["version"] == git_string
    assert version_info["tag"] == output_dict["tag"]
    assert version_info["commit"] == output_dict["commit"]
    assert version_info["commit_str"] == output_dict["commit_str"]
    assert version_info["branch"] == output_dict["branch"]
    assert version_info["dirty_flag"] == output_dict["dirty_flag"]


@pytest.mark.github_actions
@pytest.mark.parametrize(
    ("options", "config", "warn_msg"),
    [
        (["--silent"], {"silent": True}, None),
        (["--no-log-file"], {"log_file": False}, None),
        (
            ["--recalculate-all"],
            {"recalculate_beam": True, "recalculate_grid": True},
            None,
        ),
        (
            ["--snapshot-healpix-export", "--no-save-visibilities"],
            {"save_visibilities": True},
            "If we're exporting healpix we should also save the visibilities "
            "that created them. Setting save_visibilities to True",
        ),
        (
            ["--grid-uniform", "--recalculate-mapfn"],
            {"grid_uniform": True, "recalculate_mapfn": False},
            "The grid_uniform and recalculate_mapfn options are incompatible. "
            "Setting recalculate_mapfn to False.",
        ),
    ],
)
def test_configuration(options, config, warn_msg):
    """
    Test the configuration setup for pyfhd.
    This function checks if the configuration parser is correctly initialized.
    """
    sys.argv = [
        "pyfhd",
        "--config",
        str(
            importlib_resources.files("pyfhd").joinpath(
                "resources/1088285600_example/1088285600_example.yaml"
            )
        ),
        "--silent",
        "--no-log-file",
        "1088285600",
    ] + options
    # Initialize the configuration parser
    configargparser = pyfhd_parser()
    options = configargparser.parse_args()
    pyfhd_config = vars(options)
    output_dir_exists = False

    run_time = time.localtime()
    pyfhd_config, output_dir_exists = setup_directory(pyfhd_config, run_time)
    pyfhd_config = pyfhd_setup(pyfhd_config, run_time, output_dir_exists)

    # Check if the parser is an instance of ArgumentParser
    assert isinstance(configargparser, configargparse.ArgumentParser)
    assert isinstance(pyfhd_config, dict)

    config.update({"obs_id": "1088285600", "silent": True, "log_file": False})

    # TODO: add warning checking once logging fix is in.

    for key, value in config.items():
        assert pyfhd_config[key] == value

import importlib_resources
import sys
import time

import configargparse
import pytest

from pyfhd.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
from pyfhd.pyfhd import setup_directory


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

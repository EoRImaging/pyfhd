import importlib_resources
import sys
import time

import configargparse
import pytest

from pyfhd.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
from pyfhd.pyfhd import setup_directory


@pytest.mark.github_actions
def test_configuration():
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
    ]
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
    assert "obs_id" in pyfhd_config
    assert pyfhd_config["obs_id"] == "1088285600"
    assert "silent" in pyfhd_config
    assert pyfhd_config["silent"] is True
    assert "log_file" in pyfhd_config
    assert pyfhd_config["log_file"] is False
    assert "version" in pyfhd_config

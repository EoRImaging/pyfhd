from logging import Logger
from pyfhd.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
import sys
import importlib_resources
import configargparse
import pytest


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
    pyfhd_config, logger = pyfhd_setup(options)

    # Check if the parser is an instance of ArgumentParser
    assert isinstance(configargparser, configargparse.ArgumentParser)
    assert isinstance(pyfhd_config, dict)
    assert "obs_id" in pyfhd_config
    assert pyfhd_config["obs_id"] == "1088285600"
    assert "silent" in pyfhd_config
    assert pyfhd_config["silent"] is True
    assert "log_file" in pyfhd_config
    assert pyfhd_config["log_file"] is False
    assert isinstance(logger, Logger)

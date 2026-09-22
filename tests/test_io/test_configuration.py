import importlib_resources
import sys
import time

import configargparse
import pytest
from pyuvdata.testing import check_warnings

from pyfhd.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup, deprecated_args
from pyfhd.pyfhd import setup_directory


@pytest.mark.github_actions
@pytest.mark.parametrize("deprecated", [True, False])
def test_configuration(tmp_path, deprecated):
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
        "--output-path",
        str(tmp_path),
    ]

    if deprecated:
        deprecated_argv = [
            # add deprecated options to test deprecation warnings
            "--save-checkpoints",
            "--obs-checkpoint",
            "--beam-checkpoint",
            "--calibrate-checkpoint",
            "--gridding-checkpoint",
            "--save-weights",
            "--save-model",
            "--save-obs",
            "--save-params",
            "--save-cal",
            "--model-file-type",
            "uvfits",
            "--model-file-path",
            str(
                importlib_resources.files("pyfhd").joinpath(
                    "resources/1088285600_example/1088285600_model.uvfits"
                )
            ),
        ]
        sys.argv.extend(deprecated_argv)

    # Initialize the configuration parser
    configargparser = pyfhd_parser()
    options = configargparser.parse_args()
    pyfhd_config = vars(options)
    output_dir_exists = False

    run_time = time.localtime()
    pyfhd_config, output_dir_exists = setup_directory(pyfhd_config, run_time)

    if deprecated:
        pyfhd_config["cal_model_file_path"] = None
        pyfhd_config["cal_model_file_type"] = None
        warn_msgs = list(deprecated_args.values())
        warn_types = [DeprecationWarning] * len(deprecated_args)
        other_warn_msgs = [
            "model_file_path is deprecated.",
            "model_file_type is deprecated.",
            "recalculate_all is not set but setup files do not exist",
        ]
        warn_msgs.extend(other_warn_msgs)
        warn_types.extend([UserWarning] * len(other_warn_msgs))
    else:
        warn_msgs = ["recalculate_all is not set but setup files do not exist"]
        warn_types = UserWarning

    with check_warnings(warn_types, match=warn_msgs):
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

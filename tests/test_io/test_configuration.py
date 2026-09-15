from logging import Logger
from pyfhd.pyfhd_tools.pyfhd_setup import git_info, pyfhd_parser, pyfhd_setup
import sys
import importlib_resources
import configargparse
import pytest


@pytest.mark.github_actions
@pytest.mark.parametrize(
    ("git_string", "output_dict"),
    [
        (
            "1.0.3.dev321+gc96fd9451.hmf",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "branch": "hmf",
                "dirty_flag": False,
            },
        ),
        (
            "1.0.3.dev321+gc96fd9451.hmf.dirty",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "branch": "hmf",
                "dirty_flag": True,
            },
        ),
        (
            "1.0.3.dev321+gc96fd9451",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "branch": None,
                "dirty_flag": False,
            },
        ),
        (
            "1.0.3.dev321+gc96fd9451",
            {
                "tag": "1.0.3",
                "commit": "c96fd9451",
                "branch": None,
                "dirty_flag": False,
            },
        ),
        (
            "1.0.3.dev321+gc96fd9451.dirty",
            {"tag": "1.0.3", "commit": "c96fd9451", "branch": None, "dirty_flag": True},
        ),
        (
            "1.0.2",
            {"tag": "1.0.2", "commit": None, "branch": None, "dirty_flag": False},
        ),
    ],
)
def test_git_info(git_string, output_dict):
    tag, commit, branch, dirty_flag = git_info(git_string)

    assert tag == output_dict["tag"]
    assert commit == output_dict["commit"]
    assert branch == output_dict["branch"]
    assert dirty_flag == output_dict["dirty_flag"]


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
    pyfhd_config, logger = pyfhd_setup(options)

    # Check if the parser is an instance of ArgumentParser
    assert isinstance(configargparser, configargparse.ArgumentParser)
    assert isinstance(pyfhd_config, dict)
    assert isinstance(logger, Logger)

    config.update({"obs_id": "1088285600", "silent": True, "log_file": False})

    # TODO: add warning checking once logging fix is in.

    for key, value in config.items():
        assert pyfhd_config[key] == value

import logging
import sys
import time

import configargparse
import importlib_resources
import pytest
from pyuvdata import ShortDipoleBeam
from pyuvdata.testing import check_warnings

from pyfhd.io.pyfhd_io import product_file
from pyfhd.pyfhd import setup_directory
from pyfhd.pyfhd_tools.pyfhd_setup import (
    deprecated_args,
    git_info,
    pyfhd_parser,
    pyfhd_setup,
)


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
        (
            "1.0.3.dev386+gf6e2026ac.dependabot.uv.virtualenv-21.14.13",
            {
                "tag": "1.0.3",
                "commit": "f6e2026ac",
                "commit_str": "f6e2026ac (branch: dependabot.uv.virtualenv-21.14.13)",
                "branch": "dependabot.uv.virtualenv-21.14.13",
                "dirty_flag": False,
            },
        ),
        (
            "1.0.3.dev386+gf6e2026ac.dependabot.uv.virtualenv-21.14.13.dirty",
            {
                "tag": "1.0.3",
                "commit": "f6e2026ac",
                "commit_str": (
                    "f6e2026ac (branch: dependabot.uv.virtualenv-21.14.13) "
                    "DIRTY (uncommitted changes)"
                ),
                "branch": "dependabot.uv.virtualenv-21.14.13",
                "dirty_flag": True,
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
    ("options", "config", "deprecated", "warn_msgs", "make_files"),
    [
        (
            [],
            {
                "silent": True,
                "log_file": False,
                "recalculate_all": True,
                "recalculate_beam": True,
                "recalculate_cal": True,
                "recalculate_cal_model_vis": False,
                "recalculate_grid": True,
                "recalculate_healpix": True,
                "need_cal_vis": True,
                "need_beam": True,
                "need_raw_vis": True,
                "model_exists": True,
                "need_model_vis": True,
            },
            True,
            [],
            [],
        ),
        (
            [
                "--recalculate-all",
                "--cal-stop",
                "--recalculate-grid",
                "--no-save-visibilities",
                "--no-save-model-uv",
            ],
            {
                "recalculate_all": True,
                "recalculate_beam": True,
                "recalculate_cal": True,
                "recalculate_grid": False,
                "recalculate_healpix": False,
                "snapshot_healpix_export": False,
                "need_cal_vis": False,
                "need_beam": False,
                "need_raw_vis": True,
                "model_exists": True,
                "need_model_vis": True,
            },
            False,
            [
                "Both cal_stop and snapshot_healpix_export are True, but healpix "
                "cubes are not made if cal_stop is set. Setting "
                "snapshot_healpix_export to False.",
                "Both cal_stop and recalculate_grid are True, but there is no "
                "gridding if cal_stop is set. Setting recalculate_grid to False.",
                "If cal_stop is True we should save the visibilities. "
                "Setting save_visibilities to True",
                "If cal_stop is True we should save the model uv plane. "
                "Setting save_model_uv to True",
            ],
            [],
        ),
        (
            ["--recalculate-all", "--cal-stop", "--no-snapshot-healpix-export"],
            {
                "recalculate_all": True,
                "recalculate_beam": True,
                "recalculate_cal": True,
                "recalculate_grid": False,
                "recalculate_healpix": False,
                "snapshot_healpix_export": False,
                "need_cal_vis": False,
                "need_beam": False,
                "need_raw_vis": True,
                "model_exists": True,
                "need_model_vis": True,
            },
            False,
            ["The output dir"],
            ["output_dir"],
        ),
        (
            ["--description", "null"],
            {"recalculate_all": True},
            False,
            ["The output dir"],
            ["output_dir"],
        ),
        (
            ["--snapshot-healpix-export", "--no-save-visibilities"],
            {
                "recalculate_all": True,
                "recalculate_healpix": True,
                "snapshot_healpix_export": True,
                "save_visibilities": True,
            },
            False,
            [
                "If we're exporting healpix we should also save the visibilities "
                "that created them. Setting save_visibilities to True"
            ],
            [],
        ),
        (["--antenna-size", "3"], {"antenna_size": 3}, False, [], []),
        (["--cal-base-gain", "0.5"], {"cal_base_gain": 0.5}, False, [], []),
        (
            ["--beam-offset-time", "-3"],
            {"beam_offset_time": 0},
            False,
            ["You set the offset time to less than 0, it was reset to 0."],
            [],
        ),
        (
            ["--instrument", "foo"],
            {"instrument": "foo", "antenna_size": 10},
            False,
            [],
            [],
        ),
        (
            ["--beam-per-baseline", "--interpolate-kernel"],
            {"beam_per_baseline": True, "interpolate_kernel": False},
            False,
            [
                "Cannot have beam per baseline and interpolation at the same time, "
                "turning off interpolation"
            ],
            [],
        ),
        (
            ["--uvbeam-file-path", "foo"],
            {"uvbeam_file_path": None},
            False,
            [
                "Both saved_beam_file_path and uvbeam_file_path are set. "
                "Using saved_beam_file_path."
            ],
            [],
        ),
        (
            ["--analytic-beam-yaml", "foo"],
            {"analytic_beam_yaml": None},
            False,
            [
                "Both saved_beam_file_path and analytic_beam_yaml are set. "
                "Using saved_beam_file_path."
            ],
            [],
        ),
        (
            [
                "--saved-beam-file-path",
                "null",
                "--uvbeam-file-path",
                "mwa_aee_beam_files",
                "--analytic-beam-yaml",
                "foo",
            ],
            {"analytic_beam_yaml": None},
            False,
            [
                "Both uvbeam_file_path and analytic_beam_yaml are set. Using "
                "uvbeam_file_path."
            ],
            [],
        ),
        (
            [
                "--saved-beam-file-path",
                "null",
                "--uvbeam-file-path",
                "mwa_aee_beam_files",
            ],
            {"saved_beam_file_path": None},
            False,
            [],
            [],
        ),
        (
            [
                "--saved-beam-file-path",
                "null",
                "--analytic-beam-yaml",
                "!AnalyticBeam\n class: ShortDipoleBeam",
            ],
            {"analytic_beam_yaml": ShortDipoleBeam()},
            False,
            [],
            [],
        ),
        (
            ["--cal-amp-degree-fit", "3", "--no-calibration-polyfit"],
            {"cal_amp_degree_fit": 3, "calibration_polyfit": False},
            False,
            [
                "cal_amp_degree_fit and/or cal_amp_phase_fit have been set but "
                "calibration_polyfit has been disabled."
            ],
            [],
        ),
        (
            [
                "--recalculate-all",
                "--cal-reflection-hyperresolve",
                "--cal-reflection-mode-file",
            ],
            {"cal_reflection_hyperresolve": False, "cal_reflection_mode_file": True},
            False,
            [
                "cal_reflection_hyperresolve and cal_reflection_mode_file have both "
                "been turned on, cal_reflection_mode_file will be prioritised."
            ],
            [],
        ),
        (
            [
                "--cal-reflection-mode-file",
                "--cal-reflection-mode-delay",
                "--cal-reflection-mode-theory",
                "150",
            ],
            {
                "cal_reflection_mode_file": False,
                "cal_reflection_mode_delay": False,
                "cal_reflection_mode_theory": 150,
            },
            False,
            [
                "More than one nominal mode-fitting procedure specified for "
                "calibration reflection fits, prioritising cal_reflection_mode_theory"
            ],
            [],
        ),
        (
            [
                "--cal-model-file-path",
                "null",
                "--calibration-catalog-file-path",
                "gleam_cut_path",
            ],
            {
                "recalculate_all": True,
                "recalculate_cal_model_vis": True,
                "cal_model_file_path": None,
            },
            False,
            [],
            [],
        ),
        (
            [
                "--no-calibrate-visibilities",
                "--no-snapshot-healpix-export",
                "--cal-model-file-path",
                "null",
            ],
            {
                "recalculate_all": True,
                "recalculate_beam": True,
                "recalculate_cal": False,
                "recalculate_cal_model_vis": False,
                "recalculate_grid": True,
                "recalculate_healpix": False,
                "need_cal_vis": False,
                "need_beam": True,
                "need_raw_vis": True,
                "model_exists": False,
                "need_model_vis": False,
                "calibrate_visibilities": False,
            },
            False,
            [],
            [],
        ),
        (
            ["--no-recalculate-all", "--recalculate-cal-model-vis"],
            {
                "recalculate_all": False,
                "recalculate_beam": True,
                "recalculate_cal": True,
                "recalculate_cal_model_vis": False,
                "recalculate_grid": True,
                "recalculate_healpix": True,
                "need_cal_vis": True,
                "need_beam": True,
                "need_raw_vis": True,
                "model_exists": True,
                "need_model_vis": True,
            },
            False,
            [
                "recalculate_beam is not set but beam files are needed and do not "
                "exist. Recalculating beam.",
                "recalculate_cal is not set but cal files are needed and do "
                "not exist. Recalculating cal.",
                "recalculate_grid is not set but grid checkpoint files do "
                "not exist and are needed. Recalculating grid.",
                "recalculate_healpix not set but healpix checkpoint files do "
                "not exist and are needed. Recalculating healpix.",
                "recalculate_cal_model_vis is True but model_file_path is set so model "
                "visibilities do not need to be calculated. Setting "
                "recalculate_cal_model_vis to False.",
            ],
            ["obs", "params"],
        ),
        (
            [
                "--no-recalculate-all",
                "--recalculate-beam",
                "--cal-model-file-path",
                "null",
                "--calibration-catalog-file-path",
                "gleam_cut_path",
            ],
            {
                "recalculate_all": False,
                "recalculate_beam": True,
                "recalculate_cal": True,
                "recalculate_cal_model_vis": True,
                "recalculate_grid": True,
                "recalculate_healpix": True,
                "need_cal_vis": True,
                "need_beam": True,
                "need_raw_vis": True,
                "model_exists": True,
                "need_model_vis": True,
            },
            False,
            [],
            ["obs", "params"],
        ),
        (
            [
                "--no-recalculate-all",
                "--cal-model-file-path",
                "null",
                "--calibration-catalog-file-path",
                "gleam_cut_path",
            ],
            {
                "recalculate_all": False,
                "recalculate_beam": True,
                "recalculate_cal": True,
                "recalculate_cal_model_vis": True,
            },
            False,
            [
                "recalculate_cal_model_vis is not set but model vis files are needed "
                "and do not exist. Recalculating calibration model vis.",
                "recalculate_beam is not set but beam files are needed and do not "
                "exist. Recalculating beam.",
            ],
            ["obs", "params"],
        ),
        (
            [
                "--no-recalculate-all",
                "--recalculate-beam",
                "--recalculate-healpix",
                "--no-snapshot-healpix-export",
            ],
            {
                "recalculate_all": False,
                "recalculate_beam": True,
                "recalculate_cal": True,
                "recalculate_healpix": False,
            },
            False,
            [
                "recalculate_cal is not set but cal files are needed and do "
                "not exist. Recalculating cal.",
                "recalculate_healpix is True, but snapshot_healpix_export is False. "
                "Setting recalculate_healpix to False.",
            ],
            ["obs", "params"],
        ),
        (
            ["--no-recalculate-all"],
            {
                "recalculate_all": False,
                "recalculate_beam": False,
                "recalculate_cal": False,
                "recalculate_grid": True,
                "recalculate_healpix": True,
                "need_cal_vis": True,
                "need_beam": True,
                "need_raw_vis": False,
            },
            False,
            [
                "recalculate_grid is not set but grid checkpoint files do "
                "not exist and are needed. Recalculating grid.",
                "recalculate_healpix not set but healpix checkpoint files do "
                "not exist and are needed. Recalculating healpix.",
            ],
            [
                "obs",
                "params",
                "psf",
                "antenna",
                "cal",
                "cal_vis_arr",
                "cal_vis_weights",
            ],
        ),
        (
            [
                "--no-recalculate-all",
                "--recalculate-cal",
                "--no-calibrate-visibilities",
            ],
            {
                "recalculate_all": False,
                "recalculate_beam": True,
                "recalculate_cal": False,
                "recalculate_grid": True,
                "recalculate_healpix": True,
                "need_cal_vis": False,
            },
            False,
            [
                "recalculate_beam is not set but beam files are needed and do not "
                "exist. Recalculating beam.",
                "recalculate_grid is not set but grid checkpoint files do "
                "not exist and are needed. Recalculating grid.",
                "recalculate_healpix not set but healpix checkpoint files do "
                "not exist and are needed. Recalculating healpix.",
                "recalculate_cal is True but calibrate_visibilities is False. Setting "
                "recalculate_cal to False.",
            ],
            ["obs", "params"],
        ),
    ],
)
def test_configuration(
    tmp_path,
    gleam_cut_path,
    mwa_aee_beam_files,
    options,
    config,
    deprecated,
    warn_msgs,
    make_files,
):
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
    ] + options

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
    namespace = configargparser.parse_args()
    pyfhd_config = vars(namespace)

    run_time = time.localtime()
    config["log_time"] = time.strftime("%Y_%m_%d_%H_%M_%S", run_time)

    null_options = ["cal_model_file_path", "saved_beam_file_path", "description"]
    for key in null_options:
        if str(pyfhd_config[key]) == "null":
            pyfhd_config[key] = None

    if pyfhd_config["description"] is None:
        config["log_name"] = "pyfhd_" + config["log_time"]
        config["output_dir"] = tmp_path / ("pyfhd_" + config["log_time"])
    else:
        config["output_dir"] = tmp_path / (
            "pyfhd_" + pyfhd_config["description"].replace(" ", "_")
        )

    if (
        "--calibration-catalog-file-path" in options
        and str(pyfhd_config["calibration_catalog_file_path"]) == "gleam_cut_path"
    ):
        pyfhd_config["calibration_catalog_file_path"] = gleam_cut_path

    if (
        "--uvbeam-file-path" in options
        and str(pyfhd_config["uvbeam_file_path"]) == "mwa_aee_beam_files"
    ):
        pyfhd_config.update(mwa_aee_beam_files)

    if "output_dir" in make_files:
        setup_directory(pyfhd_config, run_time)
        make_files.remove("output_dir")

    pyfhd_config, output_dir_exists = setup_directory(pyfhd_config, run_time)

    for file in make_files:
        path = product_file(file, pyfhd_config=pyfhd_config, mkdir=True)
        path.touch()

    warn_types = [UserWarning] * len(warn_msgs)

    if deprecated:
        pyfhd_config["cal_model_file_path"] = None
        pyfhd_config["cal_model_file_type"] = None
        warn_msgs.extend(list(deprecated_args.values()))
        warn_types.extend([DeprecationWarning] * len(deprecated_args))
        other_warn_msgs = [
            "model_file_path is deprecated.",
            "model_file_type is deprecated.",
        ]
        warn_msgs.extend(other_warn_msgs)
        warn_types.extend([UserWarning] * len(other_warn_msgs))

    if "--recalculate-all" not in options and "--no-recalculate-all" not in options:
        warn_msgs.append("recalculate_all is not set but setup files do not exist")
        warn_types.append(UserWarning)

    with check_warnings(warn_types, match=warn_msgs):
        pyfhd_config = pyfhd_setup(pyfhd_config, run_time, output_dir_exists)

    # Check if the parser is an instance of ArgumentParser
    assert isinstance(configargparser, configargparse.ArgumentParser)
    assert isinstance(pyfhd_config, dict)

    config.update({"obs_id": "1088285600", "silent": True, "log_file": False})

    for key, value in config.items():
        assert pyfhd_config[key] == value


@pytest.mark.github_actions
@pytest.mark.parametrize(
    ("options", "err_msgs"),
    [
        (
            ["--saved-beam-file-path", "null", "--uvbeam-file-path", "foo"],
            [
                "uvbeam_file_path has been enabled with a path that doesn't exist, "
                "check the path."
            ],
        ),
        (["--psf-dim", "13"], ["If set, psf-dim must be a multiple of 2."]),
        (
            ["--input-path", "foo"],
            [
                "foo doesn't exist, please check your input path",
                "1088285600.uvfits doesn't exist, please check your input path",
                "1088285600.metafits doesn't exist, please check your input path",
            ],
        ),
        (
            ["--saved-beam-file-path", "null", "--uvbeam-zfile-path", "foo"],
            [
                "uvbeam_zfile_path is set but uvbeam_file_path is not. Please "
                "specify a uvbeam_file_path.",
                "UVBeam z file foo does not exist, please check your input path",
                "No beam file, uvbeam file or analytic beam was set.",
            ],
        ),
        (
            ["--cal-model-file-path", "null"],
            [
                "If calibrating, one of model_file_path, calibration_catalog_file_path "
                "or transfer_calibration must be set."
            ],
        ),
        (
            ["--transfer-calibration", "foo"],
            [
                "transfer_calibration has been enabled with a path that doesn't "
                "exist, check the path.",
                "transfer_calibration is not yet implemented.",
            ],
        ),
        (
            ["--healpix-inds", "foo"],
            [
                "healpix_inds has been enabled with a path that doesn't exist, "
                "check the path."
            ],
        ),
        (
            ["--cal-model-file-type", "sav"],
            [
                "You selected the model-file-path and sav, but pyfhd can't find "
                "the sav file for the model params",
                "You are missing some required files to read in the model "
                "visibilities from sav files",
            ],
        ),
    ],
)
def test_config_errors(tmp_path, caplog, mwa_aee_beam_files, options, err_msgs):
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
        "--recalculate-all",
    ] + options

    # Initialize the configuration parser
    configargparser = pyfhd_parser()
    namespace = configargparser.parse_args()
    pyfhd_config = vars(namespace)

    run_time = time.localtime()
    pyfhd_config, output_dir_exists = setup_directory(pyfhd_config, run_time)

    null_options = ["cal_model_file_path", "saved_beam_file_path", "description"]
    for key in null_options:
        if str(pyfhd_config[key]) == "null":
            pyfhd_config[key] = None

    if (
        "--uvbeam-file-path" in options
        and str(pyfhd_config["uvbeam_file_path"]) == "mwa_aee_beam_files"
    ):
        pyfhd_config.update(mwa_aee_beam_files)

    with caplog.at_level(logging.ERROR) and pytest.raises(SystemExit):
        pyfhd_config = pyfhd_setup(pyfhd_config, run_time, output_dir_exists)
    for msg in err_msgs:
        assert msg in caplog.text
    exit_err_msg = (
        f"{len(err_msgs)} errors detected, check the log above to see the errors, "
        "stopping pyfhd now"
    )
    assert exit_err_msg in caplog.text

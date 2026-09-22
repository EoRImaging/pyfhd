import argparse
import importlib_resources
import logging
import re
import sys
import shutil
import time
import warnings
from glob import glob
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import configargparse
import yaml

from pyfhd.io.pyfhd_io import checkpoint_complete

logger = logging.getLogger(__name__)

deprecated_args = {
    "save_checkpoints": (
        "save_checkpoints is deprecated and does nothing. See recalculate and "
        "save options."
    ),
    "obs_checkpoint": (
        "obs_checkpoint is deprecated and does nothing. See recalculate and "
        "save options."
    ),
    "beam_checkpoint": (
        "beam_checkpoint is deprecated and does nothing. See recalculate and "
        "save options."
    ),
    "calibrate_checkpoint": (
        "calibrate_checkpoint is deprecated and does nothing. See recalculate and "
        "save options."
    ),
    "gridding_checkpoint": (
        "gridding_checkpoint is deprecated and does nothing. See recalculate and "
        "save options."
    ),
    "save_weights": (
        "save_weights is deprecated and does nothing. See save_visibilities."
    ),
    "save_model": (
        "save_model is deprecated and does nothing. See save_visibilities and "
        "save_model_uv."
    ),
    "save_obs": ("save_obs is deprecated and does nothing (obs dict is always saved)."),
    "save_params": (
        "save_params is deprecated and does nothing (params dict is always saved)."
    ),
    "save_cal": ("save_cal is deprecated and does nothing (cal dict is always saved)."),
    "model_file_type": (
        "model_file_type is deprecated. The value set here is passed to "
        "cal_model_file_type if that is not set."
    ),
    "model_file_path": (
        "model_file_path is deprecated. The value set here is passed to "
        "cal_model_file_path if that is not set."
    ),
}


INTRO = """
    ________________________________________________________________________
    |    ooooooooo.               oooooooooooo ooooo   ooooo oooooooooo.    |
    |    8888   `Y88.             8888       8 8888    888   888     Y8b    |
    |    888   .d88' oooo    ooo  888          888     888   888      888   |
    |    888ooo88P'   `88.  .8'   888oooo8     888ooooo888   888      888   |
    |    888           `88..8'    888          888     888   888      888   |
    |    888            `888'     888          888     888   888     d88'   |
    |    o888o            .8'     o888o        o888o   o888o o888bood8P'    |
    |                 .o..P'                                                |
    |                `Y8P'                                                  |
    |_______________________________________________________________________|

    Python Fast Holographic Deconvolution

    Translated from IDL to Python as a collaboration between Astronomy Data and
    Computing Services (ADACS) and the Epoch of Reionisation (EoR) Team.

    Repository: https://github.com/EoRImaging/pyfhd

    Documentation: https://pyfhd.readthedocs.io/en/latest/
"""


class OrderedBooleanOptionalAction(argparse.BooleanOptionalAction):
    """
    OrderedBooleanOptionalAction is a custom action based on BooleanOptionalAction
    that ensures that the long options are always first in the list of options.
    More specifically, it ensures the the positive long form is first (i.e. --foo),
    and the negative (i.e. --no-foo) is second. This was needed as configargparse
    when using checking the config file and the action set in the argparse, set
    the arg passed into pyfhd by either option[0] or option[1] if the value in
    the config file was set to True or False respectively. This also allows easy
    negation of values from the configuration file with the command line if we
    need to. For example we can now set silent: true in the configuration file,
    but then set --no-silent in the command line to override the config file.

    This is also still ensures that short switches are available in pyfhd with
    the BooleanOptionalAction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        longs = [o for o in self.option_strings if o.startswith("--")]
        shorts = [o for o in self.option_strings if not o.startswith("--")]
        # put “--foo”, “--no-foo” first, then any shorts like “-s”
        self.option_strings = longs + shorts


def git_info():
    version_str = version("pyfhd")
    parts = version_str.split(".")

    if ".dev" not in version_str:
        tag = version_str
        return tag, None, None, False

    dirty_flag = False
    commit = None
    branch = None

    dev_loc = version_str.find(".dev")
    tag = version_str[:dev_loc]
    if parts[-1] == "dirty":
        dirty_flag = True

    # get commit
    for part_num, part in enumerate(parts):
        if "+g" in part:
            commit_part = part_num
            commit = part.split("+g")[-1]
            break
    branch_part = commit_part + 1
    branch_flag = False
    if not dirty_flag:
        if len(parts) > commit_part + 1:
            branch_flag = True
    else:
        if len(parts) > commit_part + 2:
            branch_flag = True
    if branch_flag:
        branch = parts[branch_part]
    else:
        branch = None

    commit_str = ""
    if commit is not None:
        commit_str += f" {commit}"
        if branch is not None:
            commit_str += f" (branch: {branch})"
        if dirty_flag:
            commit_str += " DIRTY (uncommitted changes)"
    else:
        commit_str = tag

    version_info = {
        "version": version_str,
        "tag": tag,
        "commit": commit,
        "commit_str": commit_str,
        "branch": branch,
        "dirty_flag": dirty_flag,
    }
    return version_info


def pyfhd_parser():
    """
    The pyfhd_parser configures the argparse for pyfhd

    Returns
    -------
    configargparse.ArgumentParser
        The parser for pyfhd which contains the help strings for the terminal
        and Usage section of the docs.
    """

    parser = configargparse.ArgumentParser(
        prog="pyfhd",
        description="This is the Python Fast Holographic Deconvolution package, "
        "only the observation ID (obs_id) and configuration (-c, --config) is "
        "required to start your run, but you will need to modify these arguments "
        "below to get something useful. If you don't supply a configuration file, "
        "pyfhd will use the default configuration file in the resources/config "
        "directory from the pyfhd install.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=importlib_resources.files("pyfhd.resources.config").joinpath(
            "pyfhd.yaml"
        ),
        is_config_file=True,
        help="Configuration File Path for pyfhd (By default will find pyfhd in "
        "the Python Path and use the default config file "
        "pyfhd/resources/config/pyfhd.yaml).",
    )
    # Add All the Groups
    checkpoints = parser.add_argument_group(
        "Checkpoints", "Options for recalculating vs using checkpoints."
    )
    instrument = parser.add_argument_group(
        "Instrument", "Adjust parameters specific to your instrument"
    )
    beam = parser.add_argument_group(
        "Beam Setup", "Adjust Parameters for the Beam Setup"
    )
    calibration = parser.add_argument_group(
        "Calibration", "Adjust Parameters for Calibration"
    )
    flag = parser.add_argument_group("Flagging", "Adjust Parameters for Flagging")
    gridding = parser.add_argument_group("Gridding", "Tune the Gridding in pyfhd")

    export = parser.add_argument_group(
        "Export", "Adjust the outputs of the pyfhd pipeline"
    )
    plotting = parser.add_argument_group(
        "Plotting", "Adjust the plotting of the pyfhd pipeline"
    )
    model = parser.add_argument_group("Model", "Tune the modelling in pyfhd")

    healpix = parser.add_argument_group("HEALPIX", "Adjust the HEALPIX output")

    # Version Argument
    version_info = git_info()

    version_string = (
        INTRO
        + "\n"
        + f"""
    Version: {version_info["version"]}

    Git Commit Hash: {version_info["commit_str"]}
    """
    )
    parser.add_argument("-v", "--version", action="version", version=version_string)

    # General Defaults
    parser.add_argument(
        "obs_id",
        help="The Observation ID as per the MWA file naming standards. Assumes "
        "the fits files for this observation is in the uvfits-path. obs_id and "
        "uvfits replace file_path_vis from FHD",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        help="Directory for the uvfits files and other inputs.",
        default=None,
    )
    parser.add_argument(
        "--get-sample-data",
        action="store_true",
        help="Copy sample data from pyfhd package directory to the current "
        "working directory. Will copy to an 'input' directory.",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action=OrderedBooleanOptionalAction,
        help="This pyfhd stops all output to the terminal except in the case of "
        "an error and/or exception",
    )
    parser.add_argument(
        "-l",
        "--log-file",
        action=OrderedBooleanOptionalAction,
        help="Logging in a log file is enabled by default, set to False in the "
        "config to disable logging to a file.",
    )
    parser.add_argument(
        "--conserve-memory",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Optionally split processing into chunks in various places throughout "
        "pyfhd (e.g. degridding, source DFTing and mapping function construction) "
        "to limit memory usage. See `memory-threshold` for the threshold in bytes.",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=1e9,
        help="Set a memory threshold for each chunk in bytes. By default "
        "it is set at ~1GB. Note that the full memory usage will be higher, "
        "this limits additional memory usage in various places throughout pyfhd "
        "including in degridding, source DFTing and mapping function construction.",
    )
    parser.add_argument(
        "--n-pol",
        type=int,
        default=2,
        choices=[0, 2, 4],
        help="Set number of polarizations to use (XX, YY versus XX, YY, XY, YX).",
    )
    parser.add_argument(
        "--FoV",
        "--fov",
        type=float,
        default=None,
        help="A proxy for the field of view in degrees. FoV is actually used to "
        "determine kbinsize, which will be set to !RaDeg/FoV. "
        "This means that the pixel size at phase center times dimension is "
        "approximately equal to FoV, which is not equal to the actual field of "
        "view because pixels further from the phase center are larger. "
        "If set to 0, then kbinsize determines the UV resolution.",
    )
    parser.add_argument(
        "--kbinsize",
        type=float,
        default=0.5,
        help="Size of UV pixels in wavelengths. Given a defined number of pixels "
        "in dimension, this sets the UV space extent. This will supersede degpix "
        "if dimension is also set.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2048,
        help="The number of pixels in the UV plane along the u axis. Must match "
        "the value given for elements.",
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=2048,
        help="The number of pixels in the UV plane along the v axis. Must match "
        "the value given for dimension.",
    )
    parser.add_argument(
        "--min-baseline",
        type=float,
        default=1.0,
        help="The minimum baseline length in wavelengths to include in the analysis",
    )
    parser.add_argument(
        "--deproject_w_term",
        type=float,
        default=None,
        help="Enables the function for simple_deproject_w_term and uses the "
        "parameter value for the direction value in the function",
    )

    # Checkpoints & recalculating
    checkpoints.add_argument(
        "-r",
        "--recalculate-all",
        action=OrderedBooleanOptionalAction,
        help="Forces pyfhd to recalculate all values. This will ignore values "
        "set for recalculate-grid, recalculate-beam, as it will set all of them "
        "to True",
    )
    checkpoints.add_argument(
        "--recalculate-beam",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Forces pyfhd to redo the beam setup using pyfhd's beam setup.",
    )
    checkpoints.add_argument(
        "--recalculate-cal-model-vis",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Forces pyfhd to recalculate the model visibilities (degridding).",
    )
    checkpoints.add_argument(
        "--recalculate-cal",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Forces pyfhd to recalculate the calibration. This does not force "
        "recalculating the model visibilities unless recalcuate-beam or "
        "recalculate-all are also set.",
    )
    checkpoints.add_argument(
        "-g",
        "--recalculate-grid",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Forces pyfhd to recalculate the gridding function. Replaces "
        "grid_recalculate from FHD",
    )
    checkpoints.add_argument(
        "--recalculate-healpix",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Forces pyfhd to recalculate the recalculate Healpix cubes. Replaces "
        "healpix_recalculate from FHD",
    )

    # deprecated checkpoints:
    checkpoints.add_argument(
        "--save-checkpoints",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, does nothing. See recalculate and save options.",
    )
    checkpoints.add_argument(
        "--obs-checkpoint",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, does nothing. See recalculate and save options.",
    )
    checkpoints.add_argument(
        "--beam-checkpoint",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, does nothing. See recalculate and save options.",
    )
    checkpoints.add_argument(
        "--calibrate-checkpoint",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, does nothing. See recalculate and save options.",
    )
    checkpoints.add_argument(
        "--gridding-checkpoint",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, does nothing. See recalculate and save options.",
    )

    # Instrument Group
    instrument.add_argument(
        "--instrument",
        type=str,
        default="mwa",
        choices=["mwa", "ovro-lwa", "hera", "other"],
        help="Set the instrument used for the FHD run.",
    )
    instrument.add_argument(
        "--antenna-size",
        default=None,
        type=float,
        help="The antenna size in meters. Used to set the size of the beam "
        "(gridding kernel) in uv space. If instrument is 'mwa' or 'hera' this "
        "defaults sensibly, otherwise it defaults to 10 meters, which may not be "
        "sensible.",
    )
    instrument.add_argument(
        "--override-target-phasera",
        default=None,
        type=float,
        help="RA of the target phase center, which overrides the value supplied "
        "in the metafits under the header keyword RAPHASE. If the metafits "
        "doesn't exist, it ovverides the value supplied in the uvfits under the "
        "header keyword RA",
    )
    instrument.add_argument(
        "--override-target-phasedec",
        default=None,
        type=float,
        help="dec of the target phase center, which overrides the value supplied "
        "in the metafits under the header keyword DECPHASE. If the metafits "
        "doesn't exist, it overrides the value supplied in the uvfits under the "
        "header keyword Dec.",
    )

    # Beam Setup Group
    beam.add_argument(
        "-ll",
        "--lazy-load-beam",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="pyfhd will lazy load the beam HDF5 file, allowing pyfhd to be run "
        "on much smaller systems with much less memory than FHD",
    )
    beam.add_argument(
        "--interpolate-kernel",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Use interpolation of the gridding kernel while gridding and "
        "degridding, rather than selecting the closest super-resolution kernel.",
    )
    beam.add_argument(
        "-b",
        "--saved-beam-file-path",
        type=Path,
        help="The path to an FHD written beam file (h5 or sav). "
        "Cannot be used with uvbeam-file-path or analytic-beam-yaml, "
        "one of these three should be set to specify the beam.",
    )
    beam.add_argument(
        "--uvbeam-file-path",
        type=Path,
        help="The path to a beam file readable by pyuvdata.UVBeam. Consider also "
        "setting uvbeam-freq-buffer. "
        "Cannot be used with `saved-beam-file-path` or `analytic-beam-yaml`, "
        "one of these three should be set to specify the beam.",
    )
    beam.add_argument(
        "--uvbeam-zfile-path",
        type=Path,
        default=None,
        help="The path to an MWA AEE beam Z matrix file required by pyuvdata.UVBeam "
        "for MWA AEE beams. Should only be set if  `uvbeam-file-path` is set. "
        "Cannot be used with `saved-beam-file-path` or `analytic-beam-yaml`, "
        "one of these three should be set to specify the beam.",
    )
    beam.add_argument(
        "--uvbeam-mwa-include-cross-feed-coupling",
        type=Path,
        default=True,
        help="For AEE beams only. Option to include the couplings between the "
        "different feed directions (i.e. couplings between x and y feeds). "
        "Including these couplings is more correct, but they are excluded in the "
        "default IDL FHD beam so the option is supplied to enable matching with "
        "that. Default is True.",
    )
    beam.add_argument(
        "--uvbeam-freq-buffer",
        type=float,
        default=None,
        help="Buffer around data frequency range to use when reading in the beam "
        "in Hz. Set to allow partial beam loading (when possible) to save memory. "
        "If not set (the default) the whole beam file will be read in. "
        "If set, should be at least 2 * beam frequency resolution (if set too "
        "low, beam interpolation errors can occur). "
        "We suggest setting it to 2e6 (meaning 2 MHz) for the standard MWA beam "
        "file, which has 1 MHz resolution.",
    )
    # read it in as a string to decode later. Using type=yaml.safe_load doesn't
    # work when there are multiple lines to specify the beam.
    beam.add_argument(
        "--analytic-beam-yaml",
        type=str,
        help="The yaml specifier for a pyuvdata AnalyticBeam. See the pyuvdata "
        "docs for details, must be enclosed in quotes around the entire (multiline)"
        "specification. "
        "Cannot be used with `saved-beam-file-path` or `uvbeam-file-path`, "
        "one of these three should be set to specify the beam.",
    )
    beam.add_argument(
        "--beam-offset-time",
        type=float,
        default=56,
        help="Calculate the beam at a specific time within the observation. "
        "0 seconds indicates the start of the observation, and the # of seconds "
        "in an observation indicates the end of the observation.",
    )
    beam.add_argument(
        "--beam-per-baseline",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Set to true if the beams were made with corrective phases given the "
        "baseline location, which then enables the gridding to be done per baseline",
    )
    beam.add_argument(
        "--beam-nfreq-avg",
        type=int,
        default=16,
        help="The number of fine frequency channels to calculate a beam for, "
        "using the average of the frequencies."
        "The beam is a function of frequency, and a calculation on the finest "
        "level is most correct (beam_nfreq_avg=1)."
        "However, this is computationally difficult for most machines.",
    )
    beam.add_argument(
        "--psf-dim",
        default=None,
        type=int,
        help="Controls the span of the beam in u-v space. Some defaults are 30, "
        "54 (1e6 mask with -2) or 62 (1e7 with -2).",
    )
    beam.add_argument(
        "--psf-resolution",
        default=100,
        type=int,
        help="Super-resolution factor of the psf in UV space. Values greater "
        "than 1 increase the resolution of the gridding kernel.",
    )
    beam.add_argument(
        "--beam-mask-threshold",
        default=100,
        type=int,
        help="The factor at which to clip the beam model. For example, a factor "
        "of 100 would clip the beam model at 100x down from the maximum value. "
        "This removes extraneous and uncertain modelling at low levels.",
    )
    beam.add_argument(
        "--beam-clip-floor",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Set to subtract the minimum non-zero value of the beam model from "
        "all pixels.",
    )

    # Calibration Group
    calibration.add_argument(
        "-cv",
        "--calibrate-visibilities",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Turn on the calibration of the visibilities. If turned on, "
        "calibration of the dirty, modeling, and subtraction to make a residual "
        "occurs. Otherwise, none of these occur and an uncalibrated dirty cube "
        "is output.",
    )
    calibration.add_argument(
        "--calibration-catalog-file-path",
        default=None,
        type=Path,
        help="The path to a calibration file path, must be readable by pyradiosky's "
        "SkyModel object. ",
    )
    calibration.add_argument(
        "--calibration-model-delay-filter",
        default=True,
        action=OrderedBooleanOptionalAction,
        help="Option to apply a delay filter to the model visibilities after "
        "degridding. When this is done, the bandwidth of the simulated "
        "visibilities is doubled and then reduced back to the input frequency "
        "array after filtering.",
    )
    calibration.add_argument(
        "--calibration-sidelobe-catalog-file-path",
        default=None,
        type=Path,
        help="The path to a calibration catalog file path to use for the primary "
        "beam sidelobes, must be readable by pyradiosky's SkyModel object. ",
    )
    calibration.add_argument(
        "--calibration-allow-sidelobe-sources",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Option to allow sidelobe sources when creating calibration model "
        "visibilities. Also affects the defaulting of cal-beam-threshold.",
    )
    calibration.add_argument(
        "--calibration-beam-threshold",
        type=float,
        default=None,
        help="Threshold for beam cut on sources for calibration model visibilities. "
        "Sources below the beam threshold will be cut from the skymodel to avoid "
        "sources in the nulls. Defaults to 0.05 unless allow_sidelobe_sources "
        "is True, in which case the default is 0.01.",
    )
    calibration.add_argument(
        "--calibration-restrict-sources",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Option to restrict sources to near the beam center.",
    )
    calibration.add_argument(
        "--calibration-catalog-flux-threshold",
        type=float,
        default=None,
        help="Threshold for flux values to include. These are catalog fluxes, not "
        "apparent (i.e. beam-weighted) fluxes. Can be negative, indicating an "
        "upper bound on fluxes. Default is None.",
    )
    calibration.add_argument(
        "--calibration-max-sources",
        type=int,
        default=None,
        help="Maximum number of sources to include, chosen from highest to lowest "
        "apparent (i.e. beam-weighted) flux. If a sidelobe_catalog_path is provided, "
        "sources are taken first from the main lobe catalog and then from the "
        "sidelobe catalog (if max_sources is greater than the number of sources "
        "in the main lobe catalog after the various cuts). Default is None.",
    )
    calibration.add_argument(
        "--calibration-catalog-refraction",
        type=str,
        default=None,
        help="Option for what refraction algorithm to use to account for refraction in "
        "earth's atmosphere when computing the pixel locations (and therefore "
        "when calculating beam values) for calibration sources. Allowed values "
        "are None (for no refraction correction), 'idl' to use the refraction "
        "algorithm from the IDL astrolib or 'astropy' to use astropy's refraction "
        "algorithm with temperatures and pressures estimated using the IDL "
        "astrolib algorithm. Default is None.",
    )
    calibration.add_argument(
        "--calibration-catalog-spectral-index",
        type=float,
        default=None,
        help="Spectral index to use for all sources. Overwrites the spectral index "
        "from the calibration catalog. Default is None.",
    )
    calibration.add_argument(
        "--calibration-catalog-preserve-zero-spectral-index",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Option to keep any spectral indices that are set to zero. Default "
        "is False, If False, the spectral index is reset to the mean spectral "
        "index of the catalog for any sources with zero spectral index.",
    )
    calibration.add_argument(
        "--calibration-collapse-extended-sources",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Option to replace extended source components with a single component "
        "at the flux weighted average location with a flux equal to the total flux "
        "of all the components. Default is False.",
    )
    calibration.add_argument(
        "--calibration-catalog-flatten-spectrum",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Option to flatten the spectrum by the average spectral index (calculated "
        "as a flux-weighted average). Default is False",
    )
    calibration.add_argument(
        "--bandpass-calibrate",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates a bandpass. This is an average of tiles by frequency by "
        "polarization (default), beamformer-to-LNA cable types by frequency by "
        "polarization (see cable_bandpass_fit), or over the whole season by "
        "pointing by by cable type by frequency by polarization via a read-in "
        "file (see saved_run_bp). If unset, no by-frequency bandpass is used",
    )
    calibration.add_argument(
        "--cal-bp-transfer",
        type=Path,
        default=None,
        help="Use a saved bandpass for bandpass calibration. Read in the "
        "specified file with calfits, preferred format.",
    )
    calibration.add_argument(
        "--calibration-polyfit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates a polynomial fit across the frequency band for the gain, "
        "and allows a cable reflection to be fit. The orders of the polynomial "
        "fit are determined by cal_phase_degree_fit and cal_amp_degree_fit. "
        "If unset, no polynomial fit or cable reflection fit are used.",
    )
    calibration.add_argument(
        "--auto-ratio-calibration",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates the auto ratios for cable reflections and enables "
        "global bandpass",
    )
    calibration.add_argument(
        "--cable-bandpass-fit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Average the calibration solutions across tiles within a cable "
        "grouping for the particular instrument.\n"
        "Dependency: instrument_config/<instrument>_cable_length.txt",
    )
    calibration.add_argument(
        "--cal-amp-degree-fit",
        default=2,
        type=int,
        help="The nth order of the polynomial fit over the whole band to create "
        "calibration solutions for the amplitude of the gain.\n"
        "Setting it to 0 gives a 0th order polynomial fit (one number for the "
        "whole band),\n"
        "1 gives a 1st order polynomial fit (linear fit),\n"
        "2 gives a 2nd order polynomial fit (quadratic),\n"
        "n gives nth order polynomial fit.\n"
        "Requires calibration_polyfit to be enabled.",
    )
    calibration.add_argument(
        "--cal-phase-degree-fit",
        default=1,
        type=int,
        help="The nth order of the polynomial fit over the whole band to create "
        "calibration solutions for the phase of the gain.\n"
        "Setting it to 0 gives a 0th order polynomial fit (one number for the "
        "whole band),"
        "1 gives a 1st order polynomial fit (linear fit),\n"
        "2 gives a 2nd order polynomial fit (quadratic),\n"
        "n gives nth order polynomial fit.\n"
        "Requires calibration_polyfit to be enabled.",
    )
    calibration.add_argument(
        "--cal-reflection-mode-theory",
        default=150,
        type=float,
        help="Calculate theoretical cable reflection modes given the velocity "
        "and length data stored in a config file named <instrument>_cable_length.txt. "
        "File must have a header line and at least five columns (tile index, "
        "tile name, cable length, cable velocity factor, logic on whether to "
        "fit (1) or not (0)). Can set it to positive/negative cable lengths "
        "(see cal_mode_fit) to include/exclude certain cable types.",
    )
    calibration.add_argument(
        "--cal-reflection-mode-delay",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculate cable reflection modes by Fourier transforming the "
        "residual gains, removing modes contaminated by frequency flagging, and "
        "choosing the maximum mode.",
    )
    calibration.add_argument(
        "--cal-reflection-hyperresolve",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Hyperresolve and fit residual gains using nominal reflection modes "
        "(calculated from cal_reflection_mode_delay or cal_reflection_mode_theory), "
        "producing a fine tuned mode fit, amplitude, and phase. "
        "Will be ignored if cal_reflection_mode_file is set because it is assumed "
        "that a file read-in contains mode, amp, and phase to use.",
    )
    calibration.add_argument(
        "--cal-reflection-mode-file",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Use predetermined cable reflection parameters (mode, amplitude, "
        "and phase) in the calibration solutions from a file.\n"
        "The specified format of the text file must have one header line and "
        "eleven columns:\n"
        "tile index,\n"
        "tile name,\n"
        "cable length,\n"
        "cable velocity factor,\n"
        "logic on whether to fit (1) or not (0),\n"
        "mode for X,\n"
        "amplitude for X,\n"
        "phase for X,\n"
        "mode for Y,\n"
        "amplitude for Y,\n"
        "and phase for Y. The file will be instrument_config in the input directory.",
    )
    calibration.add_argument(
        "--transfer-calibration",
        type=Path,
        help="The file path of the calibration that is to be read-in. If you "
        "give a directory, pyfhd expects there to be a file called "
        "<obs_id>_cal.hdf5 for the same observation you plan to process.",
    )
    calibration.add_argument(
        "--cal-base-gain",
        type=float,
        default=None,
        help="The relative weight to give the old calibration solution "
        "when averaging it with the new solution. Set to 1 to give equal weight. "
        "Set to 2 to give more weight to the old solution and slow down "
        "convergence. Conversely, set to 0.5 to give less weight to the old "
        "solution and speed up convergence. If use_adaptive_calibration_gain "
        "is set, the weight of the new calibration solutions will be calculated "
        "in the range cal_base_gain/2. to 1.0.",
    )
    calibration.add_argument(
        "--cal-convergence-threshold",
        type=float,
        default=1e-7,
        help="Threshold at which calibration ends. Calibration convergence is "
        "quantified by the absolute value of the fractional change in the gains "
        "over the last calibration iteration. If this quantity is less than "
        "cal_convergence_threshold then calibration terminates.",
    )
    calibration.add_argument(
        "--cal-time-average",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Performs a time average of the model/data visibilities over the "
        "time steps in the observation to reduce the number of equations that "
        "are used in the linear-least squares solver. This improves computation "
        "time, but will downweight longer baseline visibilities due to their "
        "faster phase variation.",
    )
    calibration.add_argument(
        "--calibration-auto-fit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Use the autocorrelations to calibrate. This will suffer from "
        "increased, correlated noise and bit statistic errors. However, this will "
        "save the autos as the gain in the cal structure, which can be a useful "
        "diagnostic.",
    )
    calibration.add_argument(
        "--calibration-auto-initialize",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Using autocorrelations, initialize gain values for calibration. "
        "If not set, gains will initialize to 1 or the value supplied by "
        "cal_gain_init.",
    )
    calibration.add_argument(
        "--cal-gain-init",
        default=1,
        type=int,
        help="Initial gain values for calibration. Selecting accurate inital "
        "calibration values speeds up calibration and can improve convergence. "
        "This keyword will not be used if calibration_auto_initialize is set.",
    )
    calibration.add_argument(
        "--min-cal-baseline",
        type=float,
        default=50.0,
        help="The minimum baseline length in wavelengths to be used in calibration.",
    )
    calibration.add_argument(
        "--max-cal-baseline",
        type=float,
        default=None,
        help="The maximum baseline length in wavelengths to be used in calibration. "
        "If max_baseline is smaller, it will be used instead.",
    )
    calibration.add_argument(
        "--max-cal-iter",
        default=100,
        type=int,
        help="Sets the maximum number of iterations allowed for the linear "
        "least-squares solver to converge during vis_calibrate_subroutine. "
        "Ideally, this value should be left as default unless some frequencies "
        "fail to converge within 100 iterations. Do not set this value to 5 or less.",
    )
    calibration.add_argument(
        "--cal-adaptive-calibration-gain",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Controls whether to use a Kalman Filter is used to adjust the "
        "gain used in each iteration of the calibration process.",
    )
    calibration.add_argument(
        "--cal-phase-fit-iter",
        default=4,
        type=int,
        help="Set the iteration number to begin phase calibration. Before this, "
        "phase is held fixed and only amplitude is being calibrated.",
    )
    calibration.add_argument(
        "--digital-gain-jump-polyfit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Perform polynomial fitting for the amplitude separately before "
        "and after the highband digital gain jump at 187.515E6.",
    )
    calibration.add_argument(
        "--vis-baseline-hist",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates the vis_baseline_hist dictionary containing the "
        "visibility resolution ratio average and standard deviation",
    )
    calibration.add_argument(
        "--cal-stop",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Stops the code right after calibration, and saves unflagged model "
        "visibilities along with the obs structure in a folder called cal_prerun "
        "in the pyfhd file structure.\n"
        "This allows for post-processing calibration steps like multi-day "
        "averaging, but still has all of the needed information for minimal "
        "reprocessing to get to the calibration step.\n"
        "To run a post-processing run, see keywords model_transfer and transfer_psf",
    )

    # Flagging Group
    flag.add_argument(
        "--time-cut",
        type=list,
        default=None,
        help="Seconds to cut (rounded up to next time integration step) from the "
        "beginning of the observation. Can also specify a negative time to cut off "
        "the end of the observation. Specify a vector to cut at both the start "
        "and end.",
    )
    flag.add_argument(
        "-fb",
        "--flag-basic",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Flags Frequencies and Tiles based on your configuration, params, "
        "and visibility weights. The freq_use and tile_use arrays of obs will be "
        "adjusted, and the vis_weights_arr will be put in line with the freq_use "
        "and tile_use arrays. This should almost always be True; the only time "
        "you should consider turning off basic flagging is when you're dealing "
        "with simulated visibilities and weights in pyfhd.",
    )
    flag.add_argument(
        "--flag-freq-start",
        default=None,
        type=float,
        help="Frequency in MHz to begin the observation. Flags frequencies less "
        "than it. Replaces freq_start from FHD",
    )
    flag.add_argument(
        "--flag-freq-end",
        default=None,
        type=float,
        help="Frequency in MHz to end the observation. Flags frequencies greater "
        "than it. Replaces freq_end from FHD",
    )
    flag.add_argument(
        "-ft",
        "--flag-tiles",
        default=[],
        type=list,
        action="append",
        help="A list of tile names to manually flag. I repeat, a list of tile "
        "names, NOT tile indices.",
    )
    flag.add_argument(
        "-ff",
        "--flag-frequencies",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="When set to False, pyfhd will not flag any frequencies inside of "
        "`vis_flag_basic`, `vis_weights_update`, or `vis_calibration_flag`.",
    )
    flag.add_argument(
        "-fm",
        "--flag-model",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Flag the imported model based on time offsets and the tiles. "
        "Turn off if you're dealing with an already flagged model or simulation.",
    )
    flag.add_argument(
        "-fc",
        "--flag-calibration",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Flags antennas based on calculations in vis_calibration_flag",
    )
    flag.add_argument(
        "-fcf",
        "--flag-calibration-frequencies",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="If True, frequencies are flagged based on a calibration gain of 0. "
        "If False, calibration gain for frequencies is ignored.",
    )
    flag.add_argument(
        "-fv",
        "--flag-visibilities",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Flag visibilities based on calculations in vis_flag",
    )

    # Gridding Group
    gridding.add_argument(
        "--mask-mirror-indices",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Inside baseline_grid_location, optionally exclude v-axis mirrored "
        "baselines",
    )
    gridding.add_argument(
        "--image-filter",
        default="filter_uv_uniform",
        type=str,
        choices=[
            "filter_uv_uniform"
            # the following are not implemented yet. So the code just uses
            # uniform but names the files to match the selection (so it lies)
            # commenting these out as options to prevent confusion.
            # "filter_uv_hanning",
            # "filter_uv_natural",
            # "filter_uv_radial",
            # "filter_uv_tapered_uniform",
            # "filter_uv_optimal",
        ],
        help="Weighting filter to be applied to resulting snapshot images and "
        "fits files. Replaces image_filter_fn from FHD",
    )
    gridding.add_argument(
        "--grid-spectral",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Optionally use the spectral index information to scale the "
        "uv-plane in gridding",
    )
    gridding.add_argument(
        "--grid-weights",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Grid the weights for the uv plane",
    )
    gridding.add_argument(
        "--grid-variance",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Grid the variance for the uv plane",
    )
    gridding.add_argument(
        "--grid-uniform",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Grid uniformally by applying a uniform weighted filter to all uv-planes",
    )

    # Deconvolution Group
    # deconv.add_argument(
    #     "-d",
    #     "--deconvolve",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="Run Fast Holographic Deconvolution",
    # )
    # deconv.add_argument(
    #     "--max-deconvolution-components",
    #     type=int,
    #     default=20000,
    #     help="The number of source components allowed to be found in fast "
    #     "holographic deconvolution.",
    # )
    # deconv.add_argument(
    #     "--filter-background",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="Filters out large-scale background fluctuations before "
    #     "deconvolving point sources.",
    # )
    # deconv.add_argument(
    #     "--smooth-width",
    #     default=32,
    #     type=int,
    #     help="Integer equal to the size of the region to smooth when filtering "
    #     "out large-scale background fluctuations.",
    # )
    parser.add_argument(
        "--dft-threshold",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Set to True to use the DFT approximation. When set equal to 0 the "
        "true DFT is calculated for each source. "
        "It can also be explicitly set to a value that determines the accuracy "
        "of the approximation.",
    )
    # deconv.add_argument(
    #     "--return-decon-visibilities",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="When activated degrid and export the visibilities formed from "
    #     "the deconvolution model",
    # )
    # deconv.add_argument(
    #     "--deconvolution-filter",
    #     default="filter_uv_uniform",
    #     type=str,
    #     choices=[
    #         "filter_uv_uniform",
    #         "filter_uv_hanning",
    #         "filter_uv_natural",
    #         "filter_uv_radial",
    #         "filter_uv_tapered_uniform",
    #         "filter_uv_optimal",
    #     ],
    #     help="Filter applied to images from deconvolution.",
    # )

    # Export Group
    export.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Set the output path for the current run, note a directory will "
        "still be created inside the given path",
        default=".",
    )
    export.add_argument(
        "--description",
        type=str,
        default=None,
        help="A more detailed description of the current task is applied "
        "to the output directory and logs where the output will be stored. "
        "By default, the date and time is used.",
    )
    export.add_argument(
        "--pad-uv-image",
        type=float,
        default=1.0,
        help="Pad the UV image by this factor with 0's along the outside so that "
        "output images have a higher resolution.",
    )
    export.add_argument(
        "--ring-radius-multi",
        type=float,
        default=10,
        help="Sets the multiplier for the size of the rings around sources in "
        "the restored images. "
        "Ring Radius will equal pad-uv-image * ring-radius-multi. "
        "To generate restored images without rings, set ring_radius = 0.",
    )
    export.add_argument(
        "--export-images",
        help="Export fits files and images of the sky.",
        action=OrderedBooleanOptionalAction,
        default=True,
    )
    export.add_argument(
        "--save-beam",
        default=True,
        action=OrderedBooleanOptionalAction,
        help="Save the antenna and psf dicts. These take some time to construct "
        "but also take a lot of space on disk.",
    )
    export.add_argument(
        "--save-skymodel",
        default=True,
        action=OrderedBooleanOptionalAction,
        help="Save the skymodel used to form visibilities for calibration and "
        "subtraction (if different).",
    )
    export.add_argument(
        "--save-model-uv",
        default=True,
        action=OrderedBooleanOptionalAction,
        help="Save the model uv used to form visibilities for calibration and "
        "subtraction (if different).",
    )
    export.add_argument(
        "--save-visibilities",
        default=True,
        action=OrderedBooleanOptionalAction,
        help="Save the calibrated data visibilities, the model visibilities "
        "(if calculated) and the visibility weights",
    )
    export.add_argument(
        "--save-weights",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, has no effect. See save-visibilities.",
    )
    export.add_argument(
        "--save-model",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, has no effect. See save-visibilities and save-model-uv.",
    )
    export.add_argument(
        "--save-obs",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, has no effect (obs dict is always saved).",
    )
    export.add_argument(
        "--save-params",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, has no effect (params dict is always saved).",
    )
    export.add_argument(
        "--save-cal",
        default=None,
        action=OrderedBooleanOptionalAction,
        help="Deprecated, has no effect (cal dict is always saved).",
    )
    export.add_argument(
        "--save-healpix-fits",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Create HEALPix fits files. HEALPix fits maps are in units Jy/sr. "
        "Replaces write_healpix_fits",
    )
    export.add_argument(
        "--snapshot-healpix-export",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save model, dirty, residual, weights, and variance cubes as healpix "
        "arrays, which are split into even and odd time samples, in preparation "
        "for eppsilon.",
    )

    # Plotting Group
    plotting.add_argument(
        "--calibration-plots",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Turns on the plotting of the calibration solutions",
    )
    plotting.add_argument(
        "--gridding-plots",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Turns on the plotting of the continuum gridding outputs",
    )
    plotting.add_argument(
        "--image-plots",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Turns on the plotting of the continuum fits images",
    )
    plotting.add_argument(
        "--log_plots",
        default=True,
        action=OrderedBooleanOptionalAction,
        help="Set to False to use linear scaling for images rather than log scaling.",
    )
    plotting.add_argument(
        "--sigma_clipping",
        type=int,
        default=3,
        help="The sigma level to use for sigma clipping when calculating the "
        "standard deviation of the image for image.py when using the log option.",
    )
    plotting.add_argument(
        "--percentile_clipping",
        type=int,
        default=1,
        help="The percentile level to use for percentile clipping when "
        "calculating the standard deviation of the image for image.py when using "
        "the linear option.",
    )

    # Model Group
    model.add_argument(
        "--cal-model-file-type",
        default=None,
        choices=["sav", "uvfits"],
        help="Set the file type of the model, by default it looks for sav files "
        "of format <obs_id>_params.sav and <obs_id>_vis_model_<pol_name>.sav. ",
    )
    model.add_argument(
        "--cal-model-file-path",
        default=None,
        type=Path,
        help="In the case you chose 'sav' for cal-model-file-type then this will be a "
        "directory containing all the <obs_id>_params and "
        "<obs_id>_vis_model_<pol_name> sav files. "
        "In the case you chose 'uvfits', then the path is to a uvfits file, in "
        "which case make sure the phase centre of model data must match the 'RA' "
        "and 'DEC' values in the metafits file (NOT the 'RAPHASE' and 'DECPHASE')."
        "If you want pyfhd to create model visibilities for calibration from a "
        "catalog, set this to None (~) and set calibration-catalog-file-path. "
        "To do a run without a model, this must be set to None (~),"
        "and calibrate-visibilities should be set as False. ",
    )
    model.add_argument(
        "-m",
        "--model-file-type",
        default=None,
        choices=["sav", "uvfits"],
        help="Deprecated. The value set here is passed to cal_model_file_type "
        "if that is not set.",
    )
    model.add_argument(
        "--model-file-path",
        default=None,
        type=Path,
        help="Deprecated. The value set here is passed to cal_model_file_path "
        "if that is not set.",
    )

    model.add_argument(
        "--allow-sidelobe-model-sources",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Allows pyfhd to model sources in the sidelobes for subtraction. "
        "In order to capture the sidelobe sources during the generation "
        "of a model calibration source catalog, 1 %% of the beam must be used, "
        "which is done by setting the beam_threshold to 0.01.",
    )

    # Simultation Group
    # sim.add_argument(
    #     "-sim",
    #     "--run-simulation",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="Run an in situ simulation, where model visibilities are made and "
    #     "input as the dirty visibilities (see Barry et. al. 2016 for more "
    #     "information on use-cases). "
    #     "In the case where in-situ-sim-input is not provided visibilities will "
    #     "be made within the current pyfhd run.",
    # )
    # sim.add_argument(
    #     "--in-situ-sim-input",
    #     type=Path,
    #     default=None,
    #     help="Inputs model visibilities from a previous run, which is the "
    # "preferred method since that run is independently documented.",
    # )
    # sim.add_argument(
    #     "--eor-vis-filepath",
    #     type=Path,
    #     default=None,
    #     help="A path to a file of EoR visibilities to include the EoR in the "
    #     "dirty input visibilities. in-situ-sim-input must be used in order to "
    #     "use this parameter. Replaces eor_savefile from FHD",
    # )
    # sim.add_argument(
    #     "--enhance-eor",
    #     type=float,
    #     default=1.0,
    #     help="Input a multiplicative factor to boost the signal of the EoR in "
    #     "the dirty input visibilities. in-situ-sim-input must be used in order "
    #     "to use this parameter.",
    # )
    # sim.add_argument(
    #     "--sim-noise",
    #     type=Path,
    #     default=None,
    #     help="Add a uncorrelated thermal noise to the input dirty visibilities "
    #     "from a file, or create them for the run. in-situ-sim-input must be "
    #     "used in order to use this parameter.",
    # )
    # sim.add_argument(
    #     "--remove-sim-flags",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="Bypass main flagging for in situ simulations and remove all "
    #     "weighting to remove pfb effects and flagged channels.",
    # )
    # sim.add_argument(
    #     "--tile-flag-list",
    #     type=list,
    #     help="A string array of tile names to manually flag tiles. Note that "
    #     "this is an array of tile names, not tile indices!",
    # )
    # sim.add_argument(
    #     "--extra-vis-filepath",
    #     type=Path,
    #     default=None,
    #     help="Optionally add general visibilities to the simulation, must be "
    #     "a uvfits file.",
    # )

    # HEALPIX Group
    healpix.add_argument(
        "--healpix-inds",
        default=None,
        type=Path,
        help="In the event you want to restrict the HEALPix indices to a specified "
        "file, use a combination of restrict-healpix-inds and this argument to "
        "restrict the HEALPix indexes to your given file rather than a "
        "predetermined one from the obs dictionary.",
    )
    healpix.add_argument(
        "--restrict-healpix-inds",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Only allow gridding of the output HEALPix cubes to include the "
        "HEALPix pixels specified in a file. "
        "This is useful for restricting many observations, resulting in consistent "
        "HEALPix pixels during integration. It also saves memory and walltime.",
    )
    healpix.add_argument(
        "--split-ps-export",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Split up the HEALPix outputs into even and odd time samples. "
        "This is essential to propagating errors in εppsilon. "
        "Requires more than one time sample.",
    )
    healpix.add_argument(
        "--ps-kbinsize",
        type=float,
        default=0.5,
        help="UV pixel size in wavelengths to grid for HEALPix cube generation. "
        "Overrides ps_fov and the kpix in the obs structure if set.",
    )
    healpix.add_argument(
        "--ps-fov",
        type=float,
        default=None,
        help="Field of view in degrees for HEALPix cube generation. Overrides "
        "kpix in the obs dictionary if set.",
    )
    healpix.add_argument(
        "--ps-kspan",
        type=int,
        default=0,
        help="UV plane dimension in wavelengths for HEALPix cube generation. "
        "Overrides ps_dimension and ps_degpix if set. "
        "If ps_kspan, ps_dimension, or ps_degpix are not set, the UV plane dimension "
        "is calculated from the FoV and the degpix from the obs structure.",
    )
    healpix.add_argument(
        "--ps-dimension",
        type=int,
        default=None,
        help="UV plane dimension in pixel number for HEALPix cube generation. "
        "Overrides ps_degpix if set. If ps_kspan, ps_dimension, or ps_degpix are "
        "not set, the UV plane dimension is calculated from the FoV and the "
        "degpix from the obs dictionary.",
    )
    healpix.add_argument(
        "--ps-degpix",
        type=float,
        default=None,
        help="Degrees per pixel for HEALPix cube generation. If ps_kspan, "
        "ps_dimension, or ps_degpix are not set, the UV plane dimension is "
        "calculated from the FoV and the degpix from the obs dictionary.",
    )
    healpix.add_argument(
        "--ps-nfreq-avg",
        type=float,
        default=None,
        help="A factor to average up the frequency resolution of the HEALPix "
        "cubes from the analysis frequency resolution. By default averages by a "
        "factor of 2 when this is set to None.",
    )
    healpix.add_argument(
        "--ps-beam-threshold",
        type=float,
        default=0,
        help="Minimum value to calculate the beam out to in image space. "
        "The beam in UV space is pre-calculated and may have its own "
        "beam_threshold (see that keyword for more information). This is "
        "just an additional cut in image space.",
    )
    healpix.add_argument(
        "--ps-tile-flag-list",
        type=list,
        default=[],
        action="append",
        help="A list of tile names to manually flag in the HEALPix export. "
        "I repeat, a list of tile names, NOT tile indices.",
    )
    healpix.add_argument(
        "--n-avg",
        type=int,
        default=2,
        help="Number of frequencies to average over to smooth the frequency band.",
    )
    healpix.add_argument(
        "--rephase-weights",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="If turned off, target phase center is the pointing center (as "
        "defined by Cotter). Setting to False overrides override_target_phasera "
        "and override_target_phasedec",
    )

    return parser


def _check_file_exists(config: dict, key: str) -> int:
    """
    Helper function to check if the key is not None and if it isn't None, then
    if file exists given from the config. If it does exist, replace the relative
    path with the absolute path, so when we write out paths to a config file they
    are transferable.

    Parameters
    ----------
    config : dict
        Should be the pyfhd_config
    key : str
        The keyword in the config we are checking.

    Returns
    -------
    int
        Will return 0 if there is no error, 1 if there is.
    """
    if config[key]:
        # If it doesn't exist, add error message
        test_path = Path(config[key]).expanduser().resolve()
        if not test_path.exists():
            logging.error(
                f"{key} has been enabled with a path that doesn't exist, check "
                "the path."
            )
            return 1
        # If it does exist, replace with the absolute path
        else:
            config[key] = test_path
    return 0


def write_collated_yaml_config(
    pyfhd_config: dict, output_dir: Path, description: str = ""
):
    """
    After all inputs have been validated using `pyfhd.pyfhd_tools.pyfhd_setup`,
    write out all the arguments gather in `pyfhd_config` and write out to
    a yaml configuration file. This yaml file can then be fed back into
    `pyfhd` to exactly duplicate the current run.

    Parameters
    ----------
    pyfhd_config : dict
        The options from the argparse in a dictionary
    output_dir : Path
        Path to save the file to

    """

    # for group in parser._action_groups:
    # group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    # arg_groups[group.title]=argparse.Namespace(**group_dict)

    with open(
        f"{output_dir}/{pyfhd_config['log_name']}{description}.yaml", "w"
    ) as outfile:
        outfile.write(f"# input options used for run {pyfhd_config['log_name']}\n")
        outfile.write(
            "# code version for this run: {}\n".format(pyfhd_config["version"])
        )
        outfile.write("# git hash for this run: {}\n".format(pyfhd_config["commit"]))
        outfile.write("# git branch for this run: {}\n".format(pyfhd_config["branch"]))
        outfile.write(
            "# git dirty status for this run: {}\n".format(pyfhd_config["dirty_flag"])
        )
        for key in pyfhd_config.keys():
            # These either a direct argument or are variables set internally to
            # each run,so should not appear in the yaml
            if key in [
                "top_level_dir",
                "log_name",
                "log_time",
                "commit",
                "obs_id",
                "config_file",
            ]:
                pass
            else:
                yaml_key = key.replace("_", "-")
                if pyfhd_config[key] is None:
                    outfile.write(f"{yaml_key} : ~\n")
                elif isinstance(pyfhd_config[key], float | int):
                    outfile.write(f"{yaml_key} : {pyfhd_config[key]}\n")
                elif isinstance(pyfhd_config[key], bool):
                    outfile.write(f"{yaml_key} : {pyfhd_config[key]}\n")
                # If it's a list, write it out as a list of strings
                # (Unless it's empty)
                elif isinstance(pyfhd_config[key], list):
                    if len(pyfhd_config[key]) == 0:
                        pass
                    else:
                        line = f"{key} : ["
                        line += f"'{pyfhd_config[key][0]}'"
                        for item in pyfhd_config[key][1:]:
                            line += f", '{item}'"
                        line += "]\n"
                        # for item in pyfhd_config[key]:
                        #     outfile.write(f"{key} : {item}\n")
                        outfile.write(line)
                else:
                    basic_write = False
                    try:
                        from pyuvdata.analytic_beam import AnalyticBeam

                        if isinstance(pyfhd_config[key], AnalyticBeam):
                            yaml_beam_repr = yaml.safe_dump(
                                pyfhd_config[key], default_flow_style=False
                            )
                            yaml_beam_repr = "\n  ".join(yaml_beam_repr.split("\n"))
                            outfile.write(f'{yaml_key} : "{yaml_beam_repr}"\n')
                        else:
                            basic_write = True
                    except ImportError:
                        basic_write = True

                    if basic_write:
                        outfile.write(f"{yaml_key} : '{pyfhd_config[key]}'\n")


def get_sample_data():
    """
    Copy the sample data to a folder in the current working directory.

    Parameters
    ----------
    pyfhd_config : dict
        The config dict, primarily from the yaml with a few updates.

    """
    sample_path = Path(importlib_resources.files("pyfhd")).joinpath(
        "resources/1088285600_example"
    )
    output_path = Path.cwd() / "input" / "1088285600_example"
    Path.mkdir(output_path, parents=True, exist_ok=True)

    for file in sample_path.iterdir():
        if file.is_file():
            dest_file = output_path / file.name
            if file.suffix == ".yaml":
                config = file.read_text()
                # Replace the input directory in the config with the current
                # working directory
                config = config.replace(
                    "./src/pyfhd/resources/1088285600_example", str(output_path)
                )
                dest_file.write_text(config)
                print(
                    f"Wrote the sample config file to {dest_file} with updated "
                    "paths to your machine"
                )
            else:
                shutil.copyfile(file, dest_file)
                print(f"Copied sample data file: {file.name} to {dest_file}")


def setup_directory(pyfhd_config: dict, run_time: float):
    """
    Set up the output directory and set log file name.

    Parameters
    ----------
    pyfhd_config : dict
        The configuration dictionary for pyfhd containing all the options.
    run_time : float
        The local time the run started (output of time.localtime())

    """
    log_time = time.strftime("%Y_%m_%d_%H_%M_%S", run_time)

    # define the output directory
    if pyfhd_config["description"] is None:
        dir_name = "pyfhd_" + log_time
    else:
        dir_name = "pyfhd_" + pyfhd_config["description"].replace(" ", "_")
    # Create the output directory path. If the user has selected a description,
    # don't use the time in the name - that gets used for the log
    pyfhd_config["output_path"] = (
        Path(pyfhd_config["output_path"]).expanduser().resolve()
    )
    pyfhd_config["output_dir"] = Path(pyfhd_config["output_path"], dir_name)

    if Path.is_dir(pyfhd_config["output_dir"]):
        output_dir_exists = True
    else:
        output_dir_exists = False
        Path.mkdir(pyfhd_config["output_dir"], parents=True, exist_ok=True)

    if pyfhd_config["description"] is None:
        log_name = "pyfhd_" + log_time
    else:
        log_name = (
            "pyfhd_" + pyfhd_config["description"].replace(" ", "_") + "_" + log_time
        )

    pyfhd_config["log_name"] = log_name
    pyfhd_config["log_time"] = log_time

    return pyfhd_config, output_dir_exists


@contextmanager
def pyfhd_logger(pyfhd_config: dict):
    """
    Create the logger for pyfhd.

    If silent is True in the pyfhd_config then the StreamHandler won't be added
    to logger meaning there will be no terminal output even if logger is called
    later.

    Parameters
    ----------
    pyfhd_config : dict
        The pyfhd config options dict.

    Yields
    ------
    logging.Logger
        The pyfhd logger

    """
    log = logging.getLogger("pyfhd")
    warn_log = logging.getLogger("py.warnings")  # for captureWarnings
    log.setLevel(logging.INFO)
    logging.captureWarnings(True)

    handlers = []

    if not pyfhd_config["silent"]:
        handlers.append(logging.StreamHandler())
    if pyfhd_config["log_file"]:
        handlers.append(
            logging.FileHandler(
                pyfhd_config["output_dir"] / f"{pyfhd_config['log_name']}.log"
            )
        )

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s:\n\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
        warn_log.addHandler(h)
    try:
        yield log
    finally:
        for h in handlers:
            log.removeHandler(h)
            warn_log.removeHandler(h)
            h.close()


def recalculate_flow(pyfhd_config: dict) -> dict:
    """
    Figure out what needs to be recalculated based on config and what files exist.

    Also figure out what data products we actually need to create/load during run.

    Parameters
    ----------
    pyfhd_config : dict
        The pyfhd config dict, which is updated and returned.

    Returns
    -------
    pyfhd_config : dict
        The updated config dict.

    """
    if not pyfhd_config["recalculate_all"] and not checkpoint_complete(
        "setup", pyfhd_config
    ):
        warnings.warn(
            "recalculate_all is not set but setup files do not exist. "
            "Recalculating all."
        )
        pyfhd_config["recalculate_all"] = True

    if pyfhd_config["recalculate_all"]:
        logger.info(
            "Recalculate all option has been enabled so everything will be "
            "recalculated."
        )
        pyfhd_config["recalculate_beam"] = True
        if pyfhd_config["calibrate_visibilities"]:
            pyfhd_config["recalculate_cal"] = True
            if pyfhd_config["cal_model_file_path"] is None:
                pyfhd_config["recalculate_cal_model_vis"]
        if not pyfhd_config["cal_stop"]:
            pyfhd_config["recalculate_grid"]
            if pyfhd_config["snapshot_healpix_export"]:
                pyfhd_config["recalculate_healpix"]

    if (
        pyfhd_config["recalculate_beam"]
        and pyfhd_config["calibrate_visibilities"]
        and pyfhd_config["cal_model_file_path"] is None
    ):
        pyfhd_config["recalculate_cal_model_vis"] = True

    if (
        pyfhd_config["calibrate_visibilities"]
        and pyfhd_config["cal_model_file_path"] is None
        and not pyfhd_config["recalculate_cal_model_vis"]
        and not checkpoint_complete("cal_model_vis", pyfhd_config)
    ):
        warnings.warn(
            "recalculate_cal_model_vis is not set but model vis files are needed and "
            "do not exist. Recalculating calibration model vis."
        )
        pyfhd_config["recalculate_cal_model_vis"] = True

    if (
        pyfhd_config["recalculate_cal_model_vis"]
        and pyfhd_config["calibrate_visibilities"]
    ):
        pyfhd_config["recalculate_cal"] = True

    if (
        pyfhd_config["recalculate_beam"]
        or pyfhd_config["recalculate_cal_model_vis"]
        or pyfhd_config["recalculate_cal"]
    ) and not pyfhd_config["cal_stop"]:
        pyfhd_config["recalculate_grid"] = True

        if pyfhd_config["snapshot_healpix_export"]:
            pyfhd_config["recalculate_healpix"] = True

    if (
        not pyfhd_config["recalculate_grid"]
        and not pyfhd_config["cal_stop"]
        and not checkpoint_complete("gridding", pyfhd_config)
    ):
        warnings.warn(
            "recalculate_grid is not set but grid checkpoint files do "
            "not exist and are needed. Recalculating grid."
        )
        pyfhd_config["recalculate_grid"] = True

    if (
        not pyfhd_config["recalculate_healpix"]
        and not pyfhd_config["cal_stop"]
        and pyfhd_config["snapshot_healpix_export"]
        and not checkpoint_complete("healpix", pyfhd_config)
    ):
        # N.B. This checkpoint_complete call really only checks
        # if _any_ files are there, so it's necessary but not sufficient.
        # Need more information to check for all files, deferred to main
        warnings.warn(
            "recalculate_healpix not set but healpix checkpoint files do "
            "not exist and are needed. Recalculating healpix."
        )
        pyfhd_config["recalculate_healpix"] = True

    # there are some checks below here that require a recursive call back to this
    # same function to ensure the updates flow to other recalculates as needed.
    # Make a variable to track the need for this.
    rerun = False
    # do we need calibrated visibilities?
    if pyfhd_config["calibrate_visibilities"] and (
        pyfhd_config["recalculate_grid"] or pyfhd_config["recalculate_healpix"]
    ):
        pyfhd_config["need_cal_vis"] = True
    else:
        pyfhd_config["need_cal_vis"] = False

    if (
        not pyfhd_config["recalculate_cal"]
        and pyfhd_config["need_cal_vis"]
        and not checkpoint_complete("cal", pyfhd_config)
    ):
        warnings.warn(
            "recalculate_cal is not set but cal files are needed and do "
            "not exist. Recalculating cal."
        )
        pyfhd_config["recalculate_cal"] = True
        rerun = True

    # Do we need the beam?
    if (
        pyfhd_config["recalculate_grid"]
        or pyfhd_config["recalculate_healpix"]
        or (
            pyfhd_config["recalculate_cal"]
            and pyfhd_config["cal_model_file_path"] is None
        )
    ):
        pyfhd_config["need_beam"] = True
    else:
        pyfhd_config["need_beam"] = False

    if (
        pyfhd_config["need_beam"]
        and not pyfhd_config["recalculate_beam"]
        and not checkpoint_complete("beam", pyfhd_config)
    ):
        warnings.warn(
            "recalculate_beam is not set but beam files are needed and do not "
            "exist. Recalculating beam."
        )
        pyfhd_config["recalculate_beam"] = True
        # re-run this recursively to make sure everything gets set properly
        rerun = True

    # Do we need the raw (uncalibrated) visibilities?
    if pyfhd_config["recalculate_cal"] or (
        not pyfhd_config["calibrate_visibilities"]
        and (pyfhd_config["recalculate_grid"] or pyfhd_config["recalculate_healpix"])
    ):
        pyfhd_config["need_raw_vis"] = True
    else:
        pyfhd_config["need_raw_vis"] = False

    # Does a model exist?
    # if a model was passed or if we are calibrating (and so creating one)
    if (
        pyfhd_config["calibrate_visibilities"]
        or pyfhd_config["cal_model_file_path"] is not None
    ):
        pyfhd_config["model_exists"] = True
    else:
        pyfhd_config["model_exists"] = False

    # Do we need model vis?
    # if we are calibrating or one exists and we are gridding
    if pyfhd_config["recalculate_cal"] or (
        pyfhd_config["model_exists"]
        and (pyfhd_config["recalculate_grid"] or pyfhd_config["recalculate_healpix"])
    ):
        pyfhd_config["need_model_vis"] = True
    else:
        pyfhd_config["need_model_vis"] = False

    if rerun:
        pyfhd_config = recalculate_flow(pyfhd_config)

    return pyfhd_config


def pyfhd_setup(pyfhd_config: dict, run_time: float, output_dir_exists: bool) -> dict:
    """
    Check for any incompatibilities among the options given for starting the
    pyfhd pipeline as some options do conflict with each other or have dependencies
    on other options. This function should catch all of those potential errors
    and exit the program with errors once these have been found before any output.
    This function should also replace fhd_setup

    Parameters
    ----------
    pyfhd_config : dict
        The configuration dictionary for pyfhd containing all the options.
    run_time : float
        The local time the run started (output of time.localtime())
    output_dir_exists : bool
        Boolean flag indicating whether the output directory existed before
        the run (used for warning).

    Returns
    -------
    pyfhd_config : dict
        The updated configuration dictionary for pyfhd.

    """
    stdout_time = time.strftime("%c", run_time)

    version_info = git_info()
    # add the version info to the config dict so we can carry it around as needed.
    pyfhd_config.update(version_info)

    # Show the start message
    start_string = (
        INTRO
        + "\n"
        + f"""
        Version: {pyfhd_config["version"]}

        Git Commit Hash: {pyfhd_config["commit_str"]}

        pyfhd Run Started At: {stdout_time}

        Observation ID: {pyfhd_config["obs_id"]}

        Confifuration File: {pyfhd_config["config"]}

        Validating your input...
    """
    )

    # start log
    log_string = ""
    for line in start_string.split("\n"):
        log_string += (
            line.lstrip().replace("_", " ").replace("|    ", "").replace("|", "") + "\n"
        )

    logger.info(log_string)

    # Stick a warning in the log if running in an already existing dir
    if output_dir_exists:
        warnings.warn(
            f"The output dir {pyfhd_config['output_dir']} already exists, so any "
            "existing outputs might be overridden depending on settings."
        )

    logger.info(
        "Logging and configuration file created and copied to here: {}".format(
            Path(pyfhd_config["output_dir"]).resolve()
        )
    )

    # Keep track of the errors
    n_errors = 0

    # issue deprecation warnings for deprecated arguments:
    for arg, msg in deprecated_args.items():
        if pyfhd_config[arg] is not None:
            warnings.warn(msg, DeprecationWarning)

    pyfhd_config["top_level_dir"] = str(pyfhd_config["output_dir"]).split("/")[-1]
    # Check input_path exists and obs_id uvfits and metafits files exist (Error)
    pyfhd_config["input_path"] = Path(pyfhd_config["input_path"]).expanduser().resolve()
    if not pyfhd_config["input_path"].exists():
        logger.error(
            "{} doesn't exist, please check your input path".format(
                pyfhd_config["input_path"]
            )
        )
        n_errors += 1
    obs_uvfits_path = Path(
        pyfhd_config["input_path"], pyfhd_config["obs_id"] + ".uvfits"
    )
    if not obs_uvfits_path.exists():
        logger.error(
            "{} doesn't exist, please check your input path".format(obs_uvfits_path)
        )
        n_errors += 1
    if pyfhd_config["instrument"] == "mwa":
        obs_metafits_path = Path(
            pyfhd_config["input_path"], pyfhd_config["obs_id"] + ".metafits"
        )
        if not obs_metafits_path.exists():
            logger.error(
                "{} doesn't exist, please check your input path".format(
                    obs_metafits_path
                )
            )
            n_errors += 1

    # deal with deprecated options
    if (
        pyfhd_config["cal_model_file_path"] is None
        and pyfhd_config["model_file_path"] is not None
    ):
        pyfhd_config["cal_model_file_path"] = pyfhd_config["model_file_path"]
        del pyfhd_config["model_file_path"]
        warnings.warn(
            "model_file_path is deprecated. Please use cal_model_file_path instead. "
            "Setting cal_model_file_path to what was passed to model_file_path."
        )

    if pyfhd_config["cal_model_file_path"] is not None:
        if pyfhd_config["cal_model_file_type"] is None:
            if pyfhd_config["model_file_type"] is not None:
                pyfhd_config["cal_model_file_type"] = pyfhd_config["model_file_type"]
                del pyfhd_config["model_file_type"]
                warnings.warn(
                    "model_file_type is deprecated. Please use cal_model_file_type "
                    "instead. Setting cal_model_file_type to what was passed to "
                    "model_file_type."
                )
            else:
                # sav was the old default for model_file_type, so preserve it here
                # for backwards compatibility.
                pyfhd_config["cal_model_file_type"] = "sav"
                warnings.warn(
                    "cal_model_file_path is set but cal_model_file_type is not. "
                    "Defaulting cal_model_file_type to sav."
                )

    # make recalculate options be sensible
    if (
        pyfhd_config["cal_model_file_path"] is not None
        and pyfhd_config["recalculate_cal_model_vis"]
    ):
        pyfhd_config["recalculate_cal_model_vis"] = False
        warnings.warn(
            "recalculate_cal_model_vis is True but model_file_path is set so model "
            "visibilities do not need to be calculated. Setting "
            "recalculate_cal_model_vis to False."
        )
    if not pyfhd_config["calibrate_visibilities"] and pyfhd_config["recalculate_cal"]:
        pyfhd_config["recalculate_cal"] = False
        warnings.warn(
            "recalculate_cal is True but calibrate_visibilities is False. Setting "
            "recalculate_cal to False."
        )

    if pyfhd_config["cal_stop"]:
        if pyfhd_config["recalculate_grid"]:
            pyfhd_config["recalculate_grid"] = False
            warnings.warn(
                "Both cal_stop and recalculate_grid are True, but there is no "
                "gridding if cal_stop is set. Setting recalculate_grid to False."
            )
        if pyfhd_config["snapshot_healpix_export"]:
            pyfhd_config["snapshot_healpix_export"] = False
            warnings.warn(
                "Both cal_stop and snapshot_healpix_export are True, but healpix "
                "cubes are not made if cal_stop is set. Setting "
                "snapshot_healpix_export to False."
            )
    if (
        not pyfhd_config["snapshot_healpix_export"]
        and pyfhd_config["recalculate_healpix"]
    ):
        pyfhd_config["recalculate_healpix"] = False
        warnings.warn(
            "recalculate_healpix is True, but snapshot_healpix_export is False. "
            "Setting recalculate_healpix to False."
        )

    pyfhd_config = recalculate_flow(pyfhd_config)

    # If cal_stop is set, ensure the visibilities and model uv plane are saved
    if pyfhd_config["cal_stop"]:
        if not pyfhd_config["save_visibilities"]:
            pyfhd_config["save_visibilities"] = True
            warnings.warn(
                "If cal_stop is True we should save the visibilities. "
                "Setting save_visibilities to True"
            )
        if not pyfhd_config["save_model_uv"]:
            pyfhd_config["save_model_uv"] = True
            warnings.warn(
                "If cal_stop is True we should save the model uv plane. "
                "Setting save_model_uv to True"
            )

    # If both mapping function and healpix export are on save the visibilities (Warning)
    if (
        pyfhd_config["snapshot_healpix_export"]
        and not pyfhd_config["save_visibilities"]
    ):
        pyfhd_config["save_visibilities"] = True
        warnings.warn(
            "If we're exporting healpix we should also save the visibilities "
            "that created them. Setting save_visibilities to True"
        )

    # default the antenna size if not set
    antenna_size_defaults = {"mwa": 5, "hera": 14}
    if pyfhd_config["antenna_size"] is None:
        if pyfhd_config["instrument"] in antenna_size_defaults:
            pyfhd_config["antenna_size"] = antenna_size_defaults[
                pyfhd_config["instrument"]
            ]
        else:
            pyfhd_config["antenna_size"] = 10

    # require psf_dim to be a multiple of 2
    if pyfhd_config["psf_dim"] is not None:
        if pyfhd_config["psf_dim"] % 2 != 0:
            logger.error("If set, psf-dim must be a multiple of 2.")
            n_errors += 1

    if pyfhd_config["beam_offset_time"] < 0:
        pyfhd_config["beam_offset_time"] = 0
        warnings.warn("You set the offset time to less than 0, it was reset to 0.")

    # If both beam and interp_flag leave a warning, prioritise beam_per_baseline
    if pyfhd_config["beam_per_baseline"] and pyfhd_config["interpolate_kernel"]:
        warnings.warn(
            "Cannot have beam per baseline and interpolation at the same time, "
            "turning off interpolation"
        )
        pyfhd_config["interpolate_kernel"] = False

    # If the user has set a beam file, check it exists (Error)
    if pyfhd_config["saved_beam_file_path"] is not None:
        if pyfhd_config["uvbeam_file_path"] is not None:
            warnings.warn(
                "Both saved_beam_file_path and uvbeam_file_path are set. "
                "Using saved_beam_file_path."
            )
            pyfhd_config["uvbeam_file_path"] = None
        if pyfhd_config["analytic_beam_yaml"] is not None:
            warnings.warn(
                "Both saved_beam_file_path and analytic_beam_yaml are set. "
                "Using saved_beam_file_path."
            )
            pyfhd_config["analytic_beam_yaml"] = None

        n_errors += _check_file_exists(pyfhd_config, "saved_beam_file_path")

    # If the user has set a uvbeam file, check it exists (Error)
    if pyfhd_config["uvbeam_file_path"] is not None:
        if pyfhd_config["analytic_beam_yaml"] is not None:
            warnings.warn(
                "Both uvbeam_file_path and analytic_beam_yaml are set. Using "
                "uvbeam_file_path."
            )
            pyfhd_config["analytic_beam_yaml"] = None
        n_errors += _check_file_exists(pyfhd_config, "uvbeam_file_path")

    # If the user has set a uvbeam z file, check it exists (Error)
    if pyfhd_config["uvbeam_zfile_path"] is not None:
        if pyfhd_config["uvbeam_file_path"] is None:
            logger.error(
                "uvbeam_zfile_path is set but uvbeam_file_path is not. Please "
                "specify a uvbeam_file_path."
            )
            n_errors += 1

        if not Path(pyfhd_config["uvbeam_zfile_path"]).exists():
            logger.error(
                f"UVBeam z file {pyfhd_config['uvbeam_zfile_path']} does not exist, "
                "please check your input path"
            )
            n_errors += 1

    if pyfhd_config["analytic_beam_yaml"] is not None:
        # do a little cleanup so it can be turned into an analytic beam
        temp = pyfhd_config["analytic_beam_yaml"]
        temp = temp.replace(": - ", ":\n-")
        temp = temp.replace("- ", "-")
        temp = temp.replace(": ", ":")
        temp = "\n".join(temp.split(" "))
        temp = temp.replace(":", ": ")
        temp = temp.replace("-", "- ")
        try:
            from pyuvdata.analytic_beam import AnalyticBeam  # noqa

            pyfhd_config["analytic_beam_yaml"] = yaml.safe_load(temp)
        except ImportError as ie:
            raise ImportError(
                "pyuvdata must be installed to use analytic beams"
            ) from ie

    if (
        pyfhd_config["saved_beam_file_path"] is None
        and pyfhd_config["uvbeam_file_path"] is None
        and pyfhd_config["analytic_beam_yaml"] is None
    ):
        logger.error("No beam file, uvbeam file or analytic beam was set.")

    # cal_bp_transfer when enabled should point to a file with a saved bandpass (Error)
    n_errors += _check_file_exists(pyfhd_config, "cal_bp_transfer")

    # If the user has set a calibration catalog file, check it exists (Error)
    if pyfhd_config["calibration_catalog_file_path"] is not None:
        n_errors += _check_file_exists(pyfhd_config, "calibration_catalog_file_path")

    # If the user has set a calibration catalog file, check it exists (Error)
    n_errors += _check_file_exists(
        pyfhd_config, "calibration_sidelobe_catalog_file_path"
    )

    if (
        pyfhd_config["calibrate_visibilities"]
        and pyfhd_config["calibration_catalog_file_path"] is None
        and pyfhd_config["cal_model_file_path"] is None
        and pyfhd_config["transfer_calibration"] is None
    ):
        logger.error(
            "If calibrating, one of model_file_path, calibration_catalog_file_path "
            "or transfer_calibration must be set."
        )
        n_errors += 1

    # If cal_amp_degree_fit or cal_phase_degree_fit have ben set but
    # calibration_polyfit isn't warn the user (Warning)
    if (
        pyfhd_config["cal_amp_degree_fit"]
        or pyfhd_config["cal_phase_degree_fit"]
        or pyfhd_config["cal_reflection_mode_theory"]
        or pyfhd_config["cal_reflection_mode_delay"]
    ) and not pyfhd_config["calibration_polyfit"]:
        warnings.warn(
            "cal_amp_degree_fit and/or cal_amp_phase_fit have been set but "
            "calibration_polyfit has been disabled."
        )

    # cal_reflection_hyperresolve gets ignored when cal_reflection_mode_file is
    # set (Warning)
    if (
        pyfhd_config["cal_reflection_hyperresolve"]
        and pyfhd_config["cal_reflection_mode_file"]
    ):
        logging.warning(
            "cal_reflection_hyperresolve and cal_reflection_mode_file have both "
            "been turned on, cal_reflection_mode_file will be prioritised."
        )
        pyfhd_config["cal_reflection_hyperresolve"] = False

    # cal_reflection_mode_theory and cal_reflection_mode_delay cannot be on at
    # the same time, prioritise mode_theory (Warning)
    logic_test = (
        1
        if pyfhd_config["cal_reflection_mode_file"]
        else (
            0 + 1
            if pyfhd_config["cal_reflection_mode_delay"]
            else 0 + 1
            if pyfhd_config["cal_reflection_mode_theory"]
            else 0
        )
    )
    if logic_test > 1:
        warnings.warn(
            "More than one nominal mode-fitting procedure specified for "
            "calibration reflection fits, prioritising cal_reflection_mode_theory"
        )
        pyfhd_config["cal_reflection_mode_file"] = False
        pyfhd_config["cal_reflection_mode_delay"] = False
        pyfhd_config["cal_reflection_mode_theory"] = True

    # cal_adaptive_calibration_gain impacts cal_base_gain if cal_base_gain isn't set
    if pyfhd_config["cal_base_gain"] is None:
        """
        Is set to 0.75 by default, confusingly the FHD code implies if
        use_adaptive_calibration_gain isn't active then base gain is 1.0
        However because they did this:

            IF N_Elements(use_adaptive_calibration_gain) EQ 0 THEN BEGIN
                use_adaptive_calibration_gain=0
            ENDIF
            IF N_Elements(calibration_base_gain) EQ 0 THEN BEGIN
                IF N_Elements(use_adaptive_calibration_gain) EQ 0 THEN BEGIN
                    calibration_base_gain=1. ELSE calibration_base_gain=0.75
            ENDIF

        Since use_adaptive_calibration_gain is set before the line then
        N_ELEMENTS(use_adaptive_calibration_gain) == 1 meaning base_gain is set to 0.75
        This confusingly means it isn't checking if use_adaptive_calibraton_gain
        is actually active but whether it has been set at all, small but
        significant difference.
        """
        pyfhd_config["cal_base_gain"] = 0.75

    # transfer_calibration depends on a file (Error)
    n_errors += _check_file_exists(pyfhd_config, "transfer_calibration")

    if pyfhd_config["transfer_calibration"] is not None:
        n_errors += 1
        logger.error("transfer_calibration is not yet implemented.")
    # smooth-width depends on filter_background (Warning)
    # if not pyfhd_config["filter_background"] and pyfhd_config["smooth_width"]:
    #     warnings.warn(
    #         "filter_background must be True for smooth_width to have any effect"
    #     )

    # if importing model visiblities from a uvfits file, check that file
    # exists
    if pyfhd_config["cal_model_file_path"] is not None:
        n_errors += _check_file_exists(pyfhd_config, "cal_model_file_path")

        if pyfhd_config["cal_model_file_type"] == "sav":
            # We're expecting to find a params file, then a vis_model_XX and "
            # "vis_model_YY at the very least
            if not Path.exists(
                Path(
                    pyfhd_config["cal_model_file_path"],
                    f"{pyfhd_config['obs_id']}_params.sav",
                )
            ):
                n_errors += 1
                logger.error(
                    "You selected the model-file-path and sav, but pyfhd can't find "
                    "the sav file for the model params"
                )
            files_in_model_path = glob(f"{pyfhd_config['model_file_path']}/*")
            pattern = rf".*{re.escape(pyfhd_config['obs_id'])}.*\.sav$"
            regex = re.compile(pattern)
            matching_files = [
                file_path for file_path in files_in_model_path if regex.match(file_path)
            ]
            if len(matching_files) <= 2:
                n_errors + 1
                logger.error(
                    "You are missing some required files to read in the model "
                    "visibilities from sav files, here is the list of found sav files: "
                    f"{matching_files}."
                )
            elif (
                pyfhd_config["n_pol"]
                and len(matching_files) < pyfhd_config["n_pol"] + 1
            ):
                n_errors += 1
                logger.error(
                    "You are missing files based on the number of polarizations you "
                    "have set, you should have a params file then "
                    f"{pyfhd_config['n_pol']} polarization files. Here is the "
                    f"list of found sav files: {matching_files}."
                )
            elif (
                pyfhd_config["n_pol"]
                and len(matching_files) > pyfhd_config["n_pol"] + 1
            ):
                warnings.warn(
                    "You have more files than expected for the number of polarizations "
                    f"you set, you set {pyfhd_config['n_pol']} polarizations but "
                    f"found {len(matching_files) - 1} polarization files. You can most "
                    "likely ignore this warning. Here is the list of found sav files: "
                    f"{matching_files}."
                )
            elif not pyfhd_config["n_pol"]:
                warnings.warn(
                    "Since you have told pyfhd before hand you are using 0 "
                    "polarizations and letting the uvfits header set the number "
                    "of polarizations, pyfhd will have no way to validate if the "
                    "number of savs is correct, check the list of found files "
                    f"carefully: {matching_files}. If you're sure this is fine, "
                    "ignore this warning."
                )

    # Entirety of Simulation Group depends on run-simulation (Error)
    # if not pyfhd_config["run_simulation"] and (
    #     pyfhd_config["in_situ_sim_input"]
    #     or pyfhd_config["eor_vis_filepath"]
    #     or pyfhd_config["sim_noise"]
    # ):
    #     logger.error(
    #         "run_simulation should be True if you're planning on running any "
    #         "type of simulation and therefore using in_situ_sim_input, "
    #         "eor_vis_filepath "
    #         "or sim_noise shouldn't be used when run_simulation is False"
    #     )
    #     n_errors += 1

    # in-situ-sim-input depends on a file (Error)
    # n_errors += _check_file_exists(pyfhd_config, "in_situ_sim_input")

    # eor_vis_filepath depends on a file (Error)
    # n_errors += _check_file_exists(pyfhd_config, "eor_vis_filepath")

    # enhance_eor depends on eor_vis_filepath when its not 1
    # if pyfhd_config["enhance_eor"] > 1 and pyfhd_config["eor_vis_filepath"]:
    #     logger.error(
    #         "enhance_eor is only used when importing general visibilities for "
    #         "a simulation, it should stay as 1 when eor_vis_filepath is not "
    #         "being used"
    #     )
    #     n_errors += 1

    # sim_noise depends on a file (Error)
    # n_errors += _check_file_exists(pyfhd_config, "sim_noise")

    # restrict_healpix_inds depends on a file (Error)
    if (
        pyfhd_config["healpix_inds"] is not None
        and pyfhd_config["restrict_healpix_inds"]
    ):
        n_errors += _check_file_exists(pyfhd_config, "healpix_inds")

    pyfhd_config["ring_radius"] = (
        pyfhd_config["pad_uv_image"] * pyfhd_config["ring_radius_multi"]
    )

    # --------------------------------------------------------------------------
    # Checks are finished, report any errors or warnings
    # --------------------------------------------------------------------------
    # If there are any errors exit the program.
    if n_errors:
        logger.error(
            f"{n_errors} errors detected, check the log above to see the errors, "
            "stopping pyfhd now"
        )
        sys.exit()

    logger.info("Input validated, starting pyfhd run now")

    # Create the config directory
    config_path = Path(pyfhd_config["output_dir"], "config")
    config_path.mkdir(exist_ok=True)
    write_collated_yaml_config(pyfhd_config, config_path)

    return pyfhd_config

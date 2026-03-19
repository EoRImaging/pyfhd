import argparse
import importlib_resources
import logging
import os
import re
import sys
import time
from glob import glob
from importlib.metadata import version
from pathlib import Path
from typing import Tuple

import configargparse
import yaml


class OrderedBooleanOptionalAction(argparse.BooleanOptionalAction):
    """
    OrderedBooleanOptionalAction is a custom action based on BooleanOptionalAction
    that ensures that the long options are always first in the list of options. More specifically,
    it ensures the the positive long form is first (i.e. --foo), and the negative (i.e. --no-foo) is second.
    This was needed as configargparse when using checking the config file and the action set in the argparse,
    set the arg passed into PyFHD by either option[0] or option[1] if the value in the config file was set to True or False
    respectively. This also allows easy negation of values from the configuration file with the command line if we need to.
    For example we can now set silent: true in the configuration file, but then set --no-silent in the command line to override the config file.

    This is also still ensures that short switches are available in PyFHD with the BooleanOptionalAction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        longs = [o for o in self.option_strings if o.startswith("--")]
        shorts = [o for o in self.option_strings if not o.startswith("--")]
        # put “--foo”, “--no-foo” first, then any shorts like “-s”
        self.option_strings = longs + shorts


def pyfhd_parser():
    """
    The pyfhd_parser configures the argparse for PyFHD

    Returns
    -------
    configargparse.ArgumentParser
        The parser for PyFHD which contains the help strings for the terminal and Usage section of the docs.
    """

    parser = configargparse.ArgumentParser(
        prog="PyFHD",
        description="This is the Python Fast Holographic Deconvolution package, only the observation ID (obs_id) and configuration (-c, --config) is required to start your run, but you should need to modify these arguments below to get something useful. If you don't supply a configuration file, PyFHD will use the default configuration file in the resources/config directory from the PyFHD install.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=importlib_resources.files("PyFHD.resources.config").joinpath(
            "pyfhd.yaml"
        ),
        is_config_file=True,
        help="Configuration File Path for PyFHD (By default will find PyFHD in the Python Path and use the default config file PyFHD/resources/config/pyfhd.yaml).",
    )
    # Add All the Groups
    checkpoints = parser.add_argument_group(
        "Checkpoints", "Activate checkpoints and Load up checkpoints"
    )
    instrument = parser.add_argument_group(
        "Instrument", "Adjust parameters specific to your instrument"
    )
    calibration = parser.add_argument_group(
        "Calibration", "Adjust Parameters for Calibration"
    )
    flag = parser.add_argument_group("Flagging", "Adjust Parameters for Flagging")
    beam = parser.add_argument_group(
        "Beam Setup", "Adjust Parameters for the Beam Setup"
    )
    gridding = parser.add_argument_group("Gridding", "Tune the Gridding in PyFHD")
    # Ready for deconvolution translation
    # deconv = parser.add_argument_group("Deconvolution", "Tune the Degridding in PyFHD")
    export = parser.add_argument_group(
        "Export", "Adjust the outputs of the PyFHD pipeline"
    )
    plotting = parser.add_argument_group(
        "Plotting", "Adjust the plotting of the PyFHD pipeline"
    )
    model = parser.add_argument_group("Model", "Tune the modelling in PyFHD")
    # Ready for simulation translation
    # sim = parser.add_argument_group(
    #     "Simulation", "Turn On Simulation and Tune the simulation"
    # )
    healpix = parser.add_argument_group("HEALPIX", "Adjust the HEALPIX output")

    # Version Argument
    try:
        from PyFHD.__git__ import __git_commit__, __git_branch__

        commit = __git_commit__
        branch = __git_branch__
    except ImportError:
        # If the git commit is not available, set it to unknown
        commit = "Unknown"
    version_string = f"""\
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

    Translated from IDL to Python as a collaboration between Astronomy Data and Computing Services (ADACS) and the Epoch of Reionisation (EoR) Team.

    Repository: https://github.com/EoRImaging/PyFHD

    Documentation: https://pyfhd.readthedocs.io/en/latest/

    Version: {version('PyFHD')}

    Git Commit Hash: {commit} ({branch})
    """
    parser.add_argument("-v", "--version", action="version", version=version_string)

    # General Defaults
    parser.add_argument(
        "obs_id",
        help="The Observation ID as per the MWA file naming standards. Assumes the fits files for this observation is in the uvfits-path. obs_id and uvfits replace file_path_vis from FHD",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        help="Directory for the uvfits files and other inputs, by default it looks for a directory called input in the working directory",
        default="./input/",
    )
    parser.add_argument(
        "--get-sample-data",
        action="store_true",
        help="Copy sample data from PyFHD package directory to the current working directory. Will copy to an 'input' directory.",
    )
    parser.add_argument(
        "-r",
        "--recalculate-all",
        action=OrderedBooleanOptionalAction,
        help="Forces PyFHD to recalculate all values. This will ignore values set for recalculate-grid, recalculate-beam, as it will set all of them to True",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action=OrderedBooleanOptionalAction,
        help="This PyFHD stops all output to the terminal except in the case of an error and/or exception",
    )
    parser.add_argument(
        "-l",
        "--log-file",
        action=OrderedBooleanOptionalAction,
        help="Logging in a log file is enabled by default, set to False in the config to disable logging to a file.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="mwa",
        choices=["mwa", "ovro-lwa", "hera", "other"],
        help="Set the instrument used for the FHD run.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2048,
        help="The number of pixels in the UV plane along one axis.",
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=2048,
        help="The number of pixels in the UV plane along the other axis.",
    )
    parser.add_argument(
        "--kbinsize",
        type=float,
        default=0.5,
        help="Size of UV pixels in wavelengths. Given a defined number of pixels in dimension, this sets the UV space extent. This will supersede degpix if dimension is also set.",
    )
    parser.add_argument(
        "--FoV",
        "--fov",
        type=float,
        default=None,
        help="A proxy for the field of view in degrees. FoV is actually used to determine kbinsize, which will be set to !RaDeg/FoV.\nThis means that the pixel size at phase center times dimension is approximately equal to FoV, which is not equal to the actual field of view owing to larger pixel sizes away from phase center.\nIf set to 0, then kbinsize determines the UV resolution.",
    )
    parser.add_argument(
        "--deproject_w_term",
        type=float,
        default=None,
        help="Enables the function for simple_deproject_w_term and uses the parameter value for the direction value in the function",
    )
    parser.add_argument(
        "--conserve-memory",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Optionally split many loops into chunks in the case of high memory usage.",
    )
    parser.add_argument(
        "--memory-threshold",
        type=int,
        default=1e9,
        help="Set a memory threshold for each chunk in set in bytes. By default it is set at ~100MB",
    )
    parser.add_argument(
        "--min-baseline",
        type=float,
        default=1.0,
        help="The minimum baseline length in wavelengths to include in the analysis",
    )
    parser.add_argument(
        "--n-pol",
        type=int,
        default=2,
        choices=[0, 2, 4],
        help="Set number of polarizations to use (XX, YY versus XX, YY, XY, YX).",
    )

    # Checkpoints
    checkpoints.add_argument(
        "--save-checkpoints",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Activates PyFHD's checkpointing system and saves them into the output directory",
    )
    checkpoints.add_argument(
        "--obs-checkpoint",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Load the checkpoint just after creating the observation metadata dictionary, should contain the observation metadata dictionary, uncalibrated visibility parameters, array and weights. If calibrate-checkpoint has been set, then obs-checkpoint will be ignored",
    )
    checkpoints.add_argument(
        "--calibrate-checkpoint",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Load the checkpoint after calibration containing the observation metadata dictionary with flagged tiles and frequencies, the calibration dictionary containing the gains and the calibrated visibility parameters, model, array and weights.",
    )
    checkpoints.add_argument(
        "--gridding-checkpoint",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Load the checkpoint after gridding containing the gridded uv planes for the image, weights, variance and filter, with an updated observation metadata dictionary. Should be used in conjunction with the calibrate-checkpoint option",
    )

    # Instrument Group
    instrument.add_argument(
        "--override-target-phasera",
        default=None,
        type=float,
        help="RA of the target phase center, which overrides the value supplied in the metafits under the header keyword RAPHASE. If the metafits doesn't exist, it ovverides the value supplied in the uvfits under the header keyword RA",
    )
    instrument.add_argument(
        "--override-target-phasedec",
        default=None,
        type=float,
        help="dec of the target phase center, which overrides the value supplied in the metafits under the header keyword DECPHASE. If the metafits doesn't exist, it overrides the value supplied in the uvfits under the header keyword Dec.",
    )

    # Calibration Group
    calibration.add_argument(
        "-cv",
        "--calibrate-visibilities",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Turn on the calibration of the visibilities. If turned on, calibration of the dirty, modelling, and subtraction to make a residual occurs. Otherwise, none of these occur and an uncalibrated dirty cube is output.",
    )
    calibration.add_argument(
        "--transfer-calibration",
        type=Path,
        help="The file path of a calibration to be read-in, if you give a directory PyFHD expects there to be a file called <obs_id>_cal.hdf5 using the same observation as you plan to process.",
    )
    calibration.add_argument(
        "--cal-stop",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Stops the code right after calibration, and saves unflagged model visibilities along with the obs structure in a folder called cal_prerun in the PyFHD file structure.\nThis allows for post-processing calibration steps like multi-day averaging, but still has all of the needed information for minimal reprocessing to get to the calibration step.\nTo run a post-processing run, see keywords model_transfer and transfer_psf",
    )
    calibration.add_argument(
        "--cal-convergence-threshold",
        type=float,
        default=1e-7,
        help="Threshold at which calibration ends. Calibration convergence is quantified by the absolute value of the fractional change in the gains over the last calibration iteration. If this quantity is less than cal_convergence_threshold then calibration terminates.",
    )
    calibration.add_argument(
        "--cal-adaptive-calibration-gain",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Controls whether to use a Kalman Filter to adjust the gain to use for each iteration of calculating calibration.",
    )
    calibration.add_argument(
        "--cal-base-gain",
        type=float,
        default=None,
        help="The relative weight to give the old calibration solution when averaging with the new. Set to 1. to give equal weight, to 2. to give more weight to the old solution and slow down convergence, or to 0.5 to give greater weight to the new solution and attempt to speed up convergence. If use_adaptive_calibration_gain is set, the weight of the new calibration solutions will be calculated in the range cal_base_gain/2. to 1.0",
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
        help="The maximum baseline length in wavelengths to be used in calibration. If max_baseline is smaller, it will be used instead.",
    )
    calibration.add_argument(
        "--cable-bandpass-fit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Average the calibration solutions across tiles within a cable grouping for the particular instrument.\nDependency: instrument_config/<instrument>_cable_length.txt",
    )
    calibration.add_argument(
        "--cal-bp-transfer",
        type=Path,
        default=None,
        help="Use a saved bandpass for bandpass calibration. Read in the specified file with calfits format greatly preferred.",
    )
    calibration.add_argument(
        "--calibration-polyfit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates a polynomial fit across the frequency band for the gain, and allows a cable reflection to be fit.\nThe orders of the polynomial fit are determined by cal_phase_degree_fit and cal_amp_degree_fit.\nIf unset, no polynomial fit or cable reflection fit are used.",
    )
    calibration.add_argument(
        "--cal-amp-degree-fit",
        default=2,
        type=int,
        help="The nth order of the polynomial fit over the whole band to create calibration solutions for the amplitude of the gain.\nSetting it to 0 gives a 0th order polynomial fit (one number for the whole band),\n1 gives a 1st order polynomial fit (linear fit),\n2 gives a 2nd order polynomial fit (quadratic),\nn gives nth order polynomial fit.\nRequires calibration_polyfit to be enabled.",
    )
    calibration.add_argument(
        "--cal-phase-degree-fit",
        default=1,
        type=int,
        help="The nth order of the polynomial fit over the whole band to create calibration solutions for the phase of the gain.\nSetting it to 0 gives a 0th order polynomial fit (one number for the whole band),\n1 gives a 1st order polynomial fit (linear fit),\n2 gives a 2nd order polynomial fit (quadratic),\nn gives nth order polynomial fit.\nRequires calibration_polyfit to be enabled.",
    )
    calibration.add_argument(
        "--cal-reflection-hyperresolve",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Hyperresolve and fit residual gains using nominal reflection modes (calculated from cal_reflection_mode_delay or cal_reflection_mode_theory),\nproducing a finetuned mode fit, amplitude, and phase.\nWill be ignored if cal_reflection_mode_file is set because it is assumed that a file read-in contains mode/amp/phase to use.",
    )
    calibration.add_argument(
        "--cal-reflection-mode-theory",
        default=150,
        type=float,
        help="Calculate theoretical cable reflection modes given the velocity and length data stored in a config file named <instrument>_cable_length.txt.\nFile must have a header line and at least five columns (tile index, tile name, cable length, cable velocity factor, logic on whether to fit (1) or not (0)).\nCan set it to positive/negative cable lengths (see cal_mode_fit) to include/exclude certain cable types.",
    )
    calibration.add_argument(
        "--cal-reflection-mode-delay",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculate cable reflection modes by Fourier transforming the residual gains, removing modes contaminated by frequency flagging, and choosing the maximum mode.",
    )
    calibration.add_argument(
        "--cal-reflection-mode-file",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Use predetermined cable reflection parameters (mode, amplitude, and phase) in the calibration solutions from a file.\nThe specified format of the text file must have one header line and eleven columns:\ntile index\ntile name\ncable length\ncable velocity factor\nlogic on whether to fit (1) or not (0)\nmode for X\namplitude for X\nphase for X\nmode for Y\namplitude for Y\nphase for Y. The file will be instrument_config of the input directory",
    )
    calibration.add_argument(
        "--calibration-auto-fit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Use the autocorrelations to calibrate. This will suffer from increased, correlated noise and bit statistic errors. However, this will save the autos as the gain in the cal structure, which can be a useful diagnostic.",
    )
    calibration.add_argument(
        "--calibration-auto-initialize",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="initialize gain values for calibration with the autocorrelations. If unset, gains will initialize to 1 or the value supplied by cal_gain_init",
    )
    calibration.add_argument(
        "--cal-gain-init",
        default=1,
        type=int,
        help="Initial gain values for calibration. Selecting accurate inital calibration values speeds up calibration and can improve convergence. This keyword will not be used if calibration_auto_initialize is set.",
    )
    calibration.add_argument(
        "--vis-baseline-hist",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates the vis_baseline_hist dictionary containing the visibility resolution ratio average and standard deviation",
    )
    calibration.add_argument(
        "--bandpass-calibrate",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates a bandpass.\nThis is an average of tiles by frequency by polarization (default), beamformer-to-LNA cable types by frequency by polarization (see cable_bandpass_fit),\nor over the whole season by pointing by by cable type by frequency by polarization via a read-in file (see saved_run_bp).\nIf unset, no by-frequency bandpass is used",
    )
    calibration.add_argument(
        "--cal-time-average",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Performs a time average of the model/data visibilities over the time steps in the observation to reduce the number of equations that are used in the linear-least squares solver. This improves computation time, but will downweight longer baseline visibilities due to their faster phase variation.",
    )
    calibration.add_argument(
        "--auto-ratio-calibration",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Calculates the auto ratios for cable reflections and enables global bandpass",
    )
    calibration.add_argument(
        "--digital-gain-jump-polyfit",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Perform polynomial fitting for the amplitude separately before and after the highband digital gain jump at 187.515E6.",
    )
    calibration.add_argument(
        "--cal-phase-fit-iter",
        default=4,
        type=int,
        help="Set the iteration number to begin phase calibration. Before this, phase is held fixed and only amplitude is being calibrated.",
    )
    calibration.add_argument(
        "--max-cal-iter",
        default=100,
        type=int,
        help="Sets the maximum number of iterations allowed for the linear least-squares solver to converge during vis_calibrate_subroutine. Ideally do not set this number unless you notice some of the frequencies not reaching convergence within 100 iterations and do not set this number to 5 or below.",
    )

    # Flagging Group
    flag.add_argument(
        "-fm",
        "--flag-model",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Flag the imported model based on time offsets and the tiles. Turn off if you're dealing with an already flagged model or simulation.",
    )
    flag.add_argument(
        "-fv",
        "--flag-visibilities",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Flag visibilities based on calculations in vis_flag",
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
        help="If True, flags frequencies based off 0 calibration gain, if False, ignores the calibration gain for frequencies",
    )
    flag.add_argument(
        "-fb",
        "--flag-basic",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Flags Frequencies and Tiles based on your configuration, params and the visibility weights.\nThe freq_use, tile_use arrays of obs will be adjusted and the vis_weights_arr adjusted to be in line with the freq_use and tile_use arrays.\nThis should be True always, the only time you should consider turning off basic flagging is when you're dealing with a simulated visibilities and weights in PyFHD",
    )
    flag.add_argument(
        "-ft",
        "--flag-tiles",
        default=[],
        type=list,
        action="append",
        help="A list of tile names to manually flag. I repeat, a list of tile names, NOT tile indices",
    )
    flag.add_argument(
        "-ff",
        "--flag-frequencies",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="When set to False, PyFHD will not flag any frequencies inside of `vis_flag_basic`, `vis_weights_update` or `vis_calibration_flag`.",
    )
    flag.add_argument(
        "--flag-freq-start",
        default=None,
        type=float,
        help="Frequency in MHz to begin the observation. Flags frequencies less than it. Replaces freq_start from FHD",
    )
    flag.add_argument(
        "--flag-freq-end",
        default=None,
        type=float,
        help="Frequency in MHz to end the observation. Flags frequencies greater than it. Replaces freq_end from FHD",
    )
    flag.add_argument(
        "--time-cut",
        type=list,
        default=None,
        help="Seconds to cut (rounded up to next time integration step) from the beginning of the observation. Can also specify a negative time to cut off the end of the observation. Specify a vector to cut at both the start and end.",
    )

    # Beam Setup Group
    beam.add_argument(
        "-b",
        "--beam-file-path",
        type=Path,
        help="The path to the file containing a sav or fits file",
    )
    beam.add_argument(
        "--uvbeam-file-path",
        type=Path,
        help="The path to a beam file readable by pyuvdata.UVBeam",
    )
    beam.add_argument(
        "--uvbeam-freq-buffer",
        type=float,
        default=None,
        help="Buffer around data frequency range to read in beam for in Hz. "
        "Set to allow partial beam loading (when possible) to save memory. "
        "If set, should be at least 2 * beam frequency resolution "
        "(if set too low, beam interpolation errors can occur).",
    )
    # read it in as a string to decode later. Using type=yaml.safe_load doesn't
    # work when there are multiple lines to specify the beam.
    beam.add_argument(
        "--analytic-beam-yaml",
        type=str,
        help="The yaml specifier for a pyuvdata AnalyticBeam.",
    )
    beam.add_argument(
        "-ll",
        "--lazy-load-beam",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="PyFHD will lazy load the beam HDF5 file, allowing PyFHD to be run on much smaller systems with much less memory than FHD",
    )
    beam.add_argument(
        "--recalculate-beam",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Forces PyFHD to redo the beam setup using PyFHD's beam setup.",
    )
    beam.add_argument(
        "--beam-nfreq-avg",
        type=int,
        default=16,
        help="The number of fine frequency channels to calculate a beam for, using the average of the frequencies.\nThe beam is a function of frequency, and a calculation on the finest level is most correct (beam_nfreq_avg=1).\nHowever, this is computationally difficult for most machines.",
    )
    beam.add_argument(
        "--psf-dim",
        default=54,
        type=int,
        help="Controls the span of the beam in u-v space. Some defaults are 30, 54 (1e6 mask with -2) or 62 (1e7 with -2).",
    )
    beam.add_argument(
        "--psf-resolution",
        default=100,
        type=int,
        help="Super-resolution factor of the psf in UV space. Values greater than 1 increase the resolution of the gridding kernel.",
    )
    beam.add_argument(
        "--beam-mask-threshold",
        default=100,
        type=int,
        help="The factor at which to clip the beam model. For example, a factor of 100 would clip the beam model at 100x down from the maximum value. This removes extraneous and uncertain modelling at low levels.",
    )
    beam.add_argument(
        "--beam-model-version",
        type=int,
        default=2,
        help="A number that indicates the tile beam model calculation.\nThis is dependent on the instrument, and specific calculations are carried out in <instrument>_beam_setup_gain.\nMWA range: 0, 1 (or anything else captured in the else statement), 2\nPAPER range: 1 (or anything else captured in the else statement), 2\nHERA range: 2 (or anything else captured in the else statement)",
    )
    beam.add_argument(
        "--beam-clip-floor",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Set to subtract the minimum non-zero value of the beam model from all pixels.",
    )
    beam.add_argument(
        "--interpolate-kernel",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Use interpolation of the gridding kernel while gridding and degridding, rather than selecting the closest super-resolution kernel.",
    )
    beam.add_argument(
        "--beam-per-baseline",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Set to true if the beams were made with corrective phases given the baseline location, which then enables the gridding to be done per baseline",
    )
    beam.add_argument(
        "--beam-offset-time",
        type=float,
        default=56,
        help="Calculate the beam at a specific time within the observation. 0 seconds indicates the start of the observation, and the # of seconds in an observation indicates the end of the observation.",
    )

    # Gridding Group
    gridding.add_argument(
        "-g",
        "--recalculate-grid",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Forces PyFHD to recalculate the gridding function. Replaces grid_recalculate from FHD",
    )
    gridding.add_argument(
        "--image-filter",
        default="filter_uv_uniform",
        type=str,
        choices=[
            "filter_uv_uniform",
            # the following are not implemented yet. So the code just uses
            # uniform but names the files to match the selection (so it lies)
            # commenting these out as options to prevent confusion.
            # "filter_uv_hanning",
            # "filter_uv_natural",
            # "filter_uv_radial",
            # "filter_uv_tapered_uniform",
            # "filter_uv_optimal",
        ],
        help="Weighting filter to be applied to resulting snapshot images and fits files. Replaces image_filter_fn from FHD",
    )
    gridding.add_argument(
        "--mask-mirror-indices",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Inside baseline_grid_location optionally exclude v-axis mirrored baselines",
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
    ),
    gridding.add_argument(
        "--grid-uniform",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Grid uniformally by applying a uniform weighted filter to all uv-planes",
    ),
    gridding.add_argument(
        "--grid-spectral",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Optionally use the spectral index information to scale the uv-plane in gridding",
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
    #     help="The number of source components allowed to be found in fast holographic deconvolution.",
    # )
    parser.add_argument(
        "--dft-threshold",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Set to True to use the DFT approximation. When set equal to 0 the true DFT is calculated for each source.\nIt can also be explicitly set to a value that determines the accuracy of the approximation.",
    )
    # deconv.add_argument(
    #     "--return-decon-visibilities",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="When activated degrid and export the visibilities formed from the deconvolution model",
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
    # deconv.add_argument(
    #     "--smooth-width",
    #     default=32,
    #     type=int,
    #     help="Integer equal to the size of the region to smooth when filtering out large-scale background fluctuations.",
    # )
    # deconv.add_argument(
    #     "--filter-background",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="Filters out large-scale background fluctuations before deconvolving point sources.",
    # )

    # Export Group
    export.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Set the output path for the current run, note a directory will still be created inside the given path",
        default="./output/",
    )
    export.add_argument(
        "--description",
        type=str,
        default=None,
        help="A more detailed description of the current task, will get applied to the output directory and logging where all output will be stored.\nBy default the date and time is used",
    )
    export.add_argument(
        "--export-images",
        help="Export fits files and images of the sky.",
        action=OrderedBooleanOptionalAction,
        default=True,
    )
    export.add_argument(
        "--snapshot-healpix-export",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save model/dirty/residual/weights/variance cubes as healpix arrays, split into even and odd time samples, in preparation for epsilon.",
    )
    export.add_argument(
        "--pad-uv-image",
        type=float,
        default=1.0,
        help="Pad the UV image by this factor with 0's along the outside so that output images are at a higher resolution.",
    )
    export.add_argument(
        "--ring-radius-multi",
        type=float,
        default=10,
        help="Sets the multiplier for the size of the rings around sources in the restored images.\nRing Radius will equal pad-uv-image * ring-radius-multi.\nTo generate restored images without rings, set ring_radius = 0.",
    )
    export.add_argument(
        "--save-obs",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save the obs dictionary created during PyFHD's run",
    )
    export.add_argument(
        "--save-params",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save the params dictionary created during PyFHD's run",
    )
    export.add_argument(
        "--save-cal",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save the calibration dictionary created during PyFHD's run",
    )
    export.add_argument(
        "--save-visibilities",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save the raw visibilities, calibrated data visibilities, the model visibilities, and the gridded uv planes",
    )
    export.add_argument(
        "--save-weights",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save the raw and calibrated weights from PyFHD's run",
    )
    export.add_argument(
        "--save-healpix-fits",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Create Healpix fits files. Healpix fits maps are in units Jy/sr. Replaces write_healpix_fits",
    )
    export.add_argument(
        "--save-model",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Save the model visibilities created transferred in during PyFHD's run as HDF5.",
    )

    # Plotting Group
    plotting.add_argument(
        "--calibration-plots",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Turns on the plotting of calibration solutions",
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

    # Model Group
    model.add_argument(
        "-m",
        "--model-file-type",
        default="sav",
        choices=["sav", "uvfits"],
        help="Set the file type of the model, by default it looks for sav files of format <obs_id>_params.sav and <obs_id>_vis_model_<pol_name>.sav.\nIf you set uvfits you must put set path using --import-model-uvfits.\nThis argument is required as PyFHD currently cannot produce a model.",
    )
    model.add_argument(
        "--model-file-path",
        default="./input",
        type=Path,
        help='In the case you chose sav for model-file-type then this will be a directory containing all the <obs_id>_params and <obs_id>_vis_model_<pol_name> sav files.\nIn the case you chose uvfits, then the path is to a uvfits file, in which case make sure the phase centre of model data must match the "RA" and "DEC" values in the metafits file (NOT the "RAPHASE" and "DECPHASE").',
    )
    model.add_argument(
        "--allow-sidelobe-model-sources",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Allows PyFHD to model sources in the sidelobes for subtraction.\nForces the beam_threshold to 0.01 in order to go down to 1%% of the beam to capture sidelobe sources during the generation of a model calibration source catalog for the particular observation.",
    )

    # Simultation Group
    # sim.add_argument(
    #     "-sim",
    #     "--run-simulation",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="Run an in situ simulation, where model visibilities are made and input as the dirty visibilities (see Barry et. al. 2016 for more information on use-cases).\nIn the case where in-situ-sim-input is not provided visibilities will be made within the current PyFHD run.",
    # )
    # sim.add_argument(
    #     "--in-situ-sim-input",
    #     type=Path,
    #     default=None,
    #     help="Inputs model visibilities from a previous run, which is the preferred method since that run is independently documented.",
    # )
    # sim.add_argument(
    #     "--eor-vis-filepath",
    #     type=Path,
    #     default=None,
    #     help="A path to a file of EoR visibilities to include the EoR in the dirty input visibilities. in-situ-sim-input must be used in order to use this parameter. Replaces eor_savefile from FHD",
    # )
    # sim.add_argument(
    #     "--enhance-eor",
    #     type=float,
    #     default=1.0,
    #     help="Input a multiplicative factor to boost the signal of the EoR in the dirty input visibilities. in-situ-sim-input must be used in order to use this parameter.",
    # )
    # sim.add_argument(
    #     "--sim-noise",
    #     type=Path,
    #     default=None,
    #     help="Add a uncorrelated thermal noise to the input dirty visibilities from a file, or create them for the run. in-situ-sim-input must be used in order to use this parameter.",
    # )
    # sim.add_argument(
    #     "--tile-flag-list",
    #     type=list,
    #     help="A string array of tile names to manually flag tiles. Note that this is an array of tile names, not tile indices!",
    # )
    # sim.add_argument(
    #     "--remove-sim-flags",
    #     default=False,
    #     action=OrderedBooleanOptionalAction,
    #     help="Bypass main flagging for in situ simulations and remove all weighting to remove pfb effects and flagged channels.",
    # )
    # sim.add_argument(
    #     "--extra-vis-filepath",
    #     type=Path,
    #     default=None,
    #     help="Optionally add general visibilities to the simulation, must be a uvfits file.",
    # )

    # HEALPIX Group
    healpix.add_argument(
        "--ps-kbinsize",
        type=float,
        default=0.5,
        help="UV pixel size in wavelengths to grid for Healpix cube generation. Overrides ps_fov and the kpix in the obs structure if set.",
    )
    healpix.add_argument(
        "--ps-kspan",
        type=int,
        default=0,
        help="UV plane dimension in wavelengths for Healpix cube generation.\nOverrides ps_dimension and ps_degpix if set.\nIf ps_kspan, ps_dimension, or ps_degpix are not set, the UV plane dimension is calculated from the FoV and the degpix from the obs structure.",
    )
    healpix.add_argument(
        "--ps-beam-threshold",
        type=float,
        default=0,
        help="Minimum value to which to calculate the beam out to in image space. The beam in UV space is pre-calculated and may have its own beam_threshold (see that keyword for more information), and this is only an additional cut in image space.",
    )
    healpix.add_argument(
        "--ps-fov",
        type=float,
        default=None,
        help="Field of view in degrees for Healpix cube generation. Overrides kpix in the obs dictionary if set.",
    )
    healpix.add_argument(
        "--ps-dimension",
        type=int,
        default=None,
        help="UV plane dimension in pixel number for Healpix cube generation. Overrides ps_degpix if set. If ps_kspan, ps_dimension, or ps_degpix are not set, the UV plane dimension is calculated from the FoV and the degpix from the obs dictionary.",
    )
    healpix.add_argument(
        "--ps-degpix",
        type=float,
        default=None,
        help="Degrees per pixel for Healpix cube generation. If ps_kspan, ps_dimension, or ps_degpix are not set, the UV plane dimension is calculated from the FoV and the degpix from the obs dictionary.",
    )
    healpix.add_argument(
        "--ps-nfreq-avg",
        type=float,
        default=None,
        help="A factor to average up the frequency resolution of the HEALPix cubes from the analysis frequency resolution. By default averages by a factor of 2 when this is set to None.",
    )
    healpix.add_argument(
        "--ps-tile-flag-list",
        type=list,
        default=[],
        action="append",
        help="A list of tile names to manually flag in the healpix export. I repeat, a list of tile names, NOT tile indices.",
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
        help="If turned off, target phase center is the pointing center (as defined by Cotter). Setting to False overrides override_target_phasera and override_target_phasedec",
    )
    healpix.add_argument(
        "--restrict-healpix-inds",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Only allow gridding of the output healpix cubes to include the healpix pixels specified in a file.\nThis is useful for restricting many observations to have consistent healpix pixels during integration, and saves on memory and walltime.",
    )
    healpix.add_argument(
        "--healpix-inds",
        default=None,
        type=Path,
        help="In the event you want to restrict the healpix indices to a specified file, use a combination of restrict-healpix-inds and this argument to restrict the healpix indexes to your given file rather than a predetermined one from the obs dictionary.",
    )
    healpix.add_argument(
        "--split-ps-export",
        default=False,
        action=OrderedBooleanOptionalAction,
        help="Split up the Healpix outputs into even and odd time samples.\nThis is essential to propogating errors in εppsilon.\nRequires more than one time sample.",
    )

    return parser


def _check_file_exists(config: dict, key: str) -> int:
    """
    Helper function to check if the key is not None and if it isn't None, then if file exists given from the config. If it does exist, replace the relative path
    with the absolute path, so when we write out paths to a config file they are
    transferable.

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
        if not Path(config[key]).expanduser().resolve().exists():
            logging.error(
                "{} has been enabled with a path that doesn't exist, check the path.".format(
                    key
                )
            )
            return 1
        # If it does exist, replace with the absolute path
        else:
            config[key] = Path(os.path.abspath(config[key]))
    return 0


def write_collated_yaml_config(
    pyfhd_config: dict, output_dir: Path, description: str = ""
):
    """
    After all inputs have been validated using `PyFHD.pyfhd_tools.pyfhd_setup`,
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
        outfile.write("# git hash for this run: {}\n".format(pyfhd_config["commit"]))
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
                if pyfhd_config[key] == None:
                    outfile.write(f"{yaml_key} : ~\n")
                elif type(pyfhd_config[key]) == float or type(pyfhd_config[key]) == int:
                    outfile.write(f"{yaml_key} : {pyfhd_config[key]}\n")
                elif type(pyfhd_config[key]) == bool:
                    outfile.write(f"{yaml_key} : {pyfhd_config[key]}\n")
                # If it's a list, write it out as a list of strings
                # (Unless it's empty)
                elif type(pyfhd_config[key]) == list:
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


def pyfhd_logger(pyfhd_config: dict) -> Tuple[logging.Logger, Path]:
    """
    Creates the the logger for PyFHD. If silent is True in the pyfhd_config then
    the StreamHandler won't be added to logger meaning there will be no terminal output
    even if logger is called later. If diable_log is True then the FileHandler won't be added
    to the logger preventing the creation fo the log file meaning subsequent calls to the logger
    will not add to or create a log file.

    Parameters
    ----------
    pyfhd_config : dict
        The options from the argparse in a dictionary

    Returns
    -------
    logger : logging.Logger
        The logger with the appropriate handlers added.
    output_dir : str
        Where the log and FHD outputs are being written to
    """
    # Get the time, Git commit and setup the name of the output directory
    run_time = time.localtime()
    stdout_time = time.strftime("%c", run_time)
    log_time = time.strftime("%Y_%m_%d_%H_%M_%S", run_time)

    if pyfhd_config["description"] is None:
        log_name = "pyfhd_" + log_time
    else:
        log_name = (
            "pyfhd_" + pyfhd_config["description"].replace(" ", "_") + "_" + log_time
        )

    try:
        from PyFHD.__git__ import __git_commit__, __git_branch__, __git_commit_date__

        commit = __git_commit__
        branch = __git_branch__
    except ImportError:
        # If the git commit is not available, set it to unknown
        commit = "Unknown"

    pyfhd_config["commit"] = commit
    pyfhd_config["log_name"] = log_name
    pyfhd_config["log_time"] = log_time
    # Format the starting string for logging
    start_string = f"""\
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

        Translated from IDL to Python as a collaboration between Astronomy Data and Computing Services (ADACS) and the Epoch of Reionisation (EoR) Team.

        Repository: https://github.com/EoRImaging/PyFHD

        Documentation: https://pyfhd.readthedocs.io/en/latest/

        Version: {version('PyFHD')}

        Git Commit Hash: {commit} ({branch})

        PyFHD Run Started At: {stdout_time}

        Observation ID: {pyfhd_config["obs_id"]}

        Confifuration File: {pyfhd_config["config"]}
        
        Validating your input..."""

    # Setup logging
    log_string = ""
    for line in start_string.split("\n"):
        log_string += (
            line.lstrip().replace("_", " ").replace("|    ", "").replace("|", "") + "\n"
        )
    # Start the PyFHD run
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Also capture Python Warnings and put it into the log as well
    logging.captureWarnings(True)
    # Create the logging for the temrinal
    if not pyfhd_config["silent"]:
        log_terminal = logging.StreamHandler()
        log_terminal.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(log_terminal)

    # Create the output directory path. If the user has selected a description,
    # don't use the time in the name - that gets used for the log
    if pyfhd_config["description"] is None:
        dir_name = "pyfhd_" + log_time
    else:
        dir_name = "pyfhd_" + pyfhd_config["description"].replace(" ", "_")
    if pyfhd_config["get_sample_data"]:
        output_dir = Path(pyfhd_config["output_path"])
    else:
        pyfhd_config["output_path"] = (
            Path(pyfhd_config["output_path"]).expanduser().resolve()
        )
        output_dir = Path(pyfhd_config["output_path"], dir_name)
    if Path.is_dir(output_dir):
        output_dir_exists = True
    else:
        output_dir_exists = False
        Path.mkdir(output_dir, parents=True, exist_ok=True)

    # Create the logger for the file
    if pyfhd_config["log_file"]:
        log_file = logging.FileHandler(Path(output_dir, log_name + ".log"))
        log_file.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(log_file)

    # Show that start message in the terminal and/or log file, unless both are turned off.
    logger.info(log_string)
    if not pyfhd_config["silent"]:
        log_terminal.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s:\n\t%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    if pyfhd_config["log_file"]:
        log_file.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s:\n\t%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    # Write out a config file based

    # Stick a warning in the log if running in an already existing dir
    if output_dir_exists:
        logger.warning(
            f"The output dir {output_dir} already exists, so any existing outputs might be overridden depending on settings."
        )

    logger.info(
        "Logging and configuration file created and copied to here: {}".format(
            Path(output_dir).resolve()
        )
    )

    return logger, output_dir


def pyfhd_setup(options: argparse.Namespace) -> Tuple[dict, logging.Logger]:
    """
    Check for any incompatibilities among the options given for starting the PyFHD pipeline as some options
    do conflict with each other or have dependencies on other options. This function should catch all of those
    potential errors and exit the program with errors once these have been found before any output. This function
    should also replace fhd_setup

    Parameters
    ----------
    options : argparse.Namespace
        The parsed argparse object.

    Returns
    -------
    pyfhd_config : dict
        The configuration dictionary for PyFHD containing all the options.
    logger : logging.Logger
        The logger with the appropriate handlers added.
    """
    # Keep track of the errors and warnings.
    errors = 0
    warnings = 0
    pyfhd_config = vars(options)
    # Start the logger
    logger, output_dir = pyfhd_logger(pyfhd_config)
    pyfhd_config["output_dir"] = output_dir
    pyfhd_config["top_level_dir"] = str(output_dir).split("/")[-1]
    # Check input_path exists and obs_id uvfits and metafits files exist (Error)
    pyfhd_config["input_path"] = Path(pyfhd_config["input_path"]).expanduser().resolve()
    if not pyfhd_config["input_path"].exists():
        logger.error(
            "{} doesn't exist, please check your input path".format(options.input_path)
        )
        errors += 1
    obs_uvfits_path = Path(
        pyfhd_config["input_path"], pyfhd_config["obs_id"] + ".uvfits"
    )
    if not obs_uvfits_path.exists():
        logger.error(
            "{} doesn't exist, please check your input path".format(obs_uvfits_path)
        )
        errors += 1
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
            errors += 1

    # Force PyFHD to recalculate the beam, gridding and mapping functions
    if pyfhd_config["recalculate_all"]:
        pyfhd_config["recalculate_beam"] = True
        pyfhd_config["recalculate_grid"] = True
        pyfhd_config["recalculate_mapfn"] = True
        logger.info(
            "Recalculate All option has been enabled, the beam, gridding and map function will be recalculated"
        )

    # If both mapping function and healpix export are on save the visibilities (Warning)
    if (
        pyfhd_config["snapshot_healpix_export"]
        and not pyfhd_config["save_visibilities"]
    ):
        pyfhd_config["save_visibilities"] = True
        logger.warning(
            "If we're exporting healpix we should also save the visibilities that created them. Setting save_visibilities to True"
        )
        warnings += 1

    if pyfhd_config["beam_offset_time"] < 0:
        pyfhd_config["beam_offset_time"] = 0
        logger.warning("You set the offset time to less than 0, it was reset to 0.")
        warnings += 1

    # If both beam and interp_flag leave a warning, prioritise beam_per_baseline
    if pyfhd_config["beam_per_baseline"] and pyfhd_config["interpolate_kernel"]:
        logger.warning(
            "Cannot have beam per baseline and interpolation at the same time, turning off interpolation"
        )
        pyfhd_config["interpolate_kernel"] = False

    # If the user has set a beam file, check it exists (Error)
    if pyfhd_config["beam_file_path"] is not None:
        if pyfhd_config["uvbeam_file_path"] is not None:
            logger.warning(
                "Both beam_file_path and uvbeam_file_path are set. Using beam_file_path."
            )
            pyfhd_config["uvbeam_file_path"] = None
        if pyfhd_config["analytic_beam_yaml"] is not None:
            logger.warning(
                "Both beam_file_path and analytic_beam_yaml are set. Using beam_file_path."
            )
            pyfhd_config["analytic_beam_yaml"] = None

        pyfhd_config["beam_file_path"] = (
            Path(pyfhd_config["beam_file_path"]).expanduser().resolve()
        )
        if not Path(pyfhd_config["beam_file_path"]).exists():
            logger.error(
                f"Beam file {pyfhd_config['beam_file_path']} does not exist, please check your input path"
            )
            errors += 1

    # If the user has set a uvbeam file, check it exists (Error)
    if pyfhd_config["uvbeam_file_path"] is not None:
        if pyfhd_config["analytic_beam_yaml"] is not None:
            logger.warning(
                "Both beam_file_path and analytic_beam_yaml are set. Using beam_file_path."
            )
            pyfhd_config["analytic_beam_yaml"] = None
        pyfhd_config["uvbeam_file_path"] = (
            Path(pyfhd_config["uvbeam_file_path"]).expanduser().resolve()
        )
        if not Path(pyfhd_config["uvbeam_file_path"]).exists():
            logger.error(
                f"UVBeam file {pyfhd_config['uvbeam_file_path']} does not exist, please check your input path"
            )
            errors += 1
    elif (
        pyfhd_config["analytic_beam_yaml"] is None
        and pyfhd_config["beam_file_path"] is None
        and pyfhd_config["instrument"] == "mwa"
    ):
        logger.info(
            "No beam or uvbeam file was set, and instrument is MWA. PyFHD will "
            "calculate the beam."
        )

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

    # cal_bp_transfer when enabled should point to a file with a saved bandpass (Error)
    errors += _check_file_exists(pyfhd_config, "cal_bp_transfer")

    # If cal_amp_degree_fit or cal_phase_degree_fit have ben set but calibration_polyfit isn't warn the user (Warning)
    if (
        pyfhd_config["cal_amp_degree_fit"]
        or pyfhd_config["cal_phase_degree_fit"]
        or pyfhd_config["cal_reflection_mode_theory"]
        or pyfhd_config["cal_reflection_mode_delay"]
    ) and not pyfhd_config["calibration_polyfit"]:
        logger.warning(
            "cal_amp_degree_fit and/or cal_amp_phase_fit have been set but calibration_polyfit has been disabled."
        )
        warnings += 1

    # cal_reflection_hyperresolve gets ignored when cal_reflection_mode_file is set (Warning)
    if (
        pyfhd_config["cal_reflection_hyperresolve"]
        and pyfhd_config["cal_reflection_mode_file"]
    ):
        logging.warning(
            "cal_reflection_hyperresolve and cal_reflection_mode_file have both been turned on, cal_reflection_mode_file will be prioritised."
        )
        pyfhd_config["cal_reflection_hyperresolve"] = False
        warnings += 1

    # cal_reflection_mode_theory and cal_reflection_mode_delay cannot be on at the same time, prioritise mode_theory (Warning)
    logic_test = (
        1
        if pyfhd_config["cal_reflection_mode_file"]
        else (
            0 + 1
            if pyfhd_config["cal_reflection_mode_delay"]
            else 0 + 1 if pyfhd_config["cal_reflection_mode_theory"] else 0
        )
    )
    if logic_test > 1:
        logger.warning(
            "More than one nominal mode-fitting procedure specified for calibration reflection fits, prioritising cal_reflection_mode_theory"
        )
        warnings += 1
        pyfhd_config["cal_reflection_mode_file"] = False
        pyfhd_config["cal_reflection_mode_delay"] = False
        pyfhd_config["cal_reflection_mode_theory"] = True

    # cal_adaptive_calibration_gain impacts cal_base_gain if cal_base_gain isn't set
    if pyfhd_config["cal_base_gain"] == None:
        """
        Is set to 0.75 by default, confusingly the FHD code implies if
        use_adaptive_calibration_gain isn't active then base gain is 1.0
        However because they did this:

            IF N_Elements(use_adaptive_calibration_gain) EQ 0 THEN use_adaptive_calibration_gain=0
            IF N_Elements(calibration_base_gain) EQ 0 THEN BEGIN
                IF N_Elements(use_adaptive_calibration_gain) EQ 0 THEN calibration_base_gain=1. ELSE calibration_base_gain=0.75

        Since use_adaptive_calibration_gain is set before the line then
        N_ELEMENTS(use_adaptive_calibration_gain) == 1 meaning base_gain is set to 0.75
        This confusingly means it isn't checking if use_adaptive_calibraton_gain is actually active
        but whether it has been set at all, small but significant difference.
        """
        pyfhd_config["cal_base_gain"] = 0.75

    # calibration_catalog_file_path depends on a file (Error)
    # errors += _check_file_exists(pyfhd_config, "calibration_catalog_file_path")

    # transfer_calibration depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, "transfer_calibration")

    # smooth-width depends on filter_background (Warning)
    # if not pyfhd_config["filter_background"] and pyfhd_config["smooth_width"]:
    #     logger.warning(
    #         "filter_background must be True for smooth_width to have any effect"
    #     )
    #     warnings += 1

    # allow_sidelobe_model_sources depends on model_visibilities (Error)
    if (
        pyfhd_config["allow_sidelobe_model_sources"]
        and not pyfhd_config["model_visibilities"]
    ):
        logger.error(
            "allow_sidelobe_model_sources shouldn't be True when model_visibilities is not, check if you meant to turn on model_visibilities"
        )
        errors += 1

    # if importing model visiblities from a uvfits file, check that file
    # exists
    if pyfhd_config["model_file_path"]:
        pyfhd_config["model_file_path"] = (
            Path(pyfhd_config["model_file_path"]).expanduser().resolve()
        )
        errors += _check_file_exists(pyfhd_config, "model_file_path")

    if pyfhd_config["model_file_path"] == "sav":
        # We're expecting to find a params file, then a vis_model_XX and vis_model_YY at the very least
        if not Path.exists(
            Path(
                pyfhd_config["model_file_path"], f"{pyfhd_config['obs_id']}_params.sav"
            )
        ):
            errors += 1
            logger.error(
                "You selected the model-file-path and sav, but PyFHD can't find the sav file for the model params"
            )
        files_in_model_path = glob(f"{pyfhd_config['model_file_path']}/*")
        pattern = rf".*{re.escape(pyfhd_config['obs_id'])}.*\.sav$"
        regex = re.compile(pattern)
        matching_files = [
            file_path for file_path in files_in_model_path if regex.match(file_path)
        ]
        if len(matching_files) <= 2:
            errors + 1
            logger.error(
                f"You are missing some required files to read in the model visibilities from sav files, here is the list of found sav files: {matching_files}."
            )
        elif pyfhd_config["n_pol"] and len(matching_files) < pyfhd_config["n_pol"] + 1:
            errors += 1
            logger.error(
                f"You are missing files based on the number of polarizations you have set, you should have a params file then {pyfhd_config['n_pol']} polarization files. Here is the list of found sav files: {matching_files}."
            )
        elif pyfhd_config["n_pol"] and len(matching_files) > pyfhd_config["n_pol"] + 1:
            warnings += 1
            logger.warning(
                f"You have more files than expected for the number of polarizations you set, you set {pyfhd_config['n_pol']} polarizations but found {len(matching_files)- 1} polarization files. You can most likely ignore this warning. Here is the list of found sav files: {matching_files}."
            )
        elif not pyfhd_config["n_pol"]:
            warnings += 1
            logger.warning(
                f"Since you have told PyFHD before hand you are using 0 polarizations and letting the uvfits header set the number of polarizations PyFHD have no way to validate if the number of savs is correct, check the list of found files carefully: {matching_files}. If you're sure this is fine, ignore this warning."
            )

    # Entirety of Simulation Group depends on run-simulation (Error)
    # if not pyfhd_config["run_simulation"] and (
    #     pyfhd_config["in_situ_sim_input"]
    #     or pyfhd_config["eor_vis_filepath"]
    #     or pyfhd_config["sim_noise"]
    # ):
    #     logger.error(
    #         "run_simulation should be True if you're planning on running any type of simulation and therefore using in_situ_sim_input, eor_vis_filepath or sim_noise shouldn't be used when run_simulation is False"
    #     )
    #     errors += 1

    # in-situ-sim-input depends on a file (Error)
    # errors += _check_file_exists(pyfhd_config, "in_situ_sim_input")

    # eor_vis_filepath depends on a file (Error)
    # errors += _check_file_exists(pyfhd_config, "eor_vis_filepath")

    # enhance_eor depends on eor_vis_filepath when its not 1
    # if pyfhd_config["enhance_eor"] > 1 and pyfhd_config["eor_vis_filepath"]:
    #     logger.error(
    #         "enhance_eor is only used when importing general visibilities for a simulation, it should stay as 1 when eor_vis_filepath is not being used"
    #     )
    #     errors += 1

    # sim_noise depends on a file (Error)
    # errors += _check_file_exists(pyfhd_config, "sim_noise")

    # restrict_healpix_inds depends on a file (Error)
    if (
        pyfhd_config["healpix_inds"] is not None
        and pyfhd_config["restrict_healpix_inds"]
    ):
        errors += _check_file_exists(pyfhd_config, "healpix_inds")

    pyfhd_config["ring_radius"] = (
        pyfhd_config["pad_uv_image"] * pyfhd_config["ring_radius_multi"]
    )

    # --------------------------------------------------------------------------
    # Checks are finished, report any errors or warnings
    # --------------------------------------------------------------------------
    # If there are any errors exit the program.
    if errors:
        logger.error(
            "{} errors detected, check the log above to see the errors, stopping PyFHD now".format(
                errors
            )
        )
        # Close the handlers in the log
        for handler in logger.handlers:
            handler.close()
        sys.exit()

    if warnings:
        logger.warning(
            "{} warnings detected, check the log above, these may cause some weird behavior".format(
                warnings
            )
        )
    logger.info("Input validated, starting PyFHD run now")

    # Create the config directory
    if not pyfhd_config["get_sample_data"]:
        config_path = Path(output_dir, "config")
        config_path.mkdir(exist_ok=True)
        write_collated_yaml_config(pyfhd_config, config_path)

    return pyfhd_config, logger

import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import h5py
from h5py import File
import numpy as np

from .beam_setup.beam import create_psf
from .calibration.calibrate import calibrate, calibrate_qu_mixing
from .data_setup.obs import create_obs
from .data_setup.uvfits import (
    create_layout,
    create_params,
    extract_header,
    extract_visibilities,
)
from .flagging.flagging import vis_flag, vis_flag_basic
from .gridding.gridding_utils import crosspol_reformat
from .gridding.visibility_grid import visibility_grid
from .pyfhd_tools.pyfhd_setup import (
    pyfhd_parser,
    pyfhd_logger,
    pyfhd_setup,
    setup_directory,
    write_collated_yaml_config,
)
from .pyfhd_tools.pyfhd_utils import (
    simple_deproject_w_term,
    vis_noise_calc,
    vis_weights_update,
)
from .source_modeling.vis_model_transfer import vis_model_transfer
from .io.pyfhd_io import save, load
from .io.pyfhd_quickview import quickview
from .healpix.export import healpix_snapshot_cube_generate
from .plotting.gridding import plot_gridding

logger = logging.getLogger(__name__)


def _print_time_diff(start: float, end: float, description: str):
    """
    Print the time difference in a nice format between start and end time

    Parameters
    ----------
    start : float
        Start time in seconds since epoch
    end : float
        End time in seconds since epoch
    """
    runtime = end - start
    if runtime > 60:
        runtime = timedelta(seconds=end - start)
        logger.info(f"{description} completed in: {runtime}")
    elif runtime < 1:
        logger.info(
            f"{description} completed in: {round(runtime * 1000, 5)} milliseconds"
        )
    else:
        logger.info(f"{description} completed in: {round(runtime, 5)} seconds")


def _finish_pyfhd(pyfhd_start: float, psf: dict | File, pyfhd_config: dict):
    pyfhd_end = time.time()
    runtime = timedelta(seconds=pyfhd_end - pyfhd_start)
    # Close all open h5 files
    if isinstance(psf, h5py.File):
        psf.close()

    # Write a final collated yaml for the final pyfhd_config
    write_collated_yaml_config(
        pyfhd_config, Path(pyfhd_config["output_dir"], "config"), "-final"
    )
    # Save the config in a HDF5 file for ease of reading in previous
    # parameters from previous runs
    save(
        Path(pyfhd_config["output_dir"], "config", "pyfhd_config.h5"),
        pyfhd_config,
        "pyfhd_config",
    )
    logger.info(
        f"pyfhd Run Completed for {pyfhd_config['obs_id']}\nTotal Runtime "
        f"(Days:Hours:Minutes:Seconds.Millseconds): {runtime}"
    )

    return


def run_pyfhd(pyfhd_config: dict, pyfhd_start: float):
    """
    Do a full pyfhd run.

    This should be called from the `main` function to ensure all the
    directories and logging are set up properly.

    Parameters
    ----------
    pyfhd_config : dict
        The config dict, primarily from the yaml with a few updates.
    pyfhd_start : float
        The run start time (for logging -- output of time.time())

    """
    pyfhd_successful = False
    try:
        if (
            pyfhd_config["description"] is not None
            and pyfhd_config["description"] != ""
        ):
            checkpoint_name = pyfhd_config["description"] + "_" + pyfhd_config["obs_id"]
        else:
            checkpoint_name = pyfhd_config["obs_id"]

        if pyfhd_config["save_checkpoints"]:
            pyfhd_config["checkpoint_dir"] = Path(
                pyfhd_config["output_dir"], "checkpoints"
            )
            pyfhd_config["checkpoint_dir"].mkdir(exist_ok=True)

            obs_checkpoint_file = Path(
                pyfhd_config["checkpoint_dir"], f"{checkpoint_name}_obs_checkpoint.h5"
            )
            if pyfhd_config["obs_checkpoint"] and not obs_checkpoint_file.exists():
                logger.warning(
                    "obs_checkpoint is set but obs checkpoint file does not exist. "
                    "Recalculating obs."
                )
                pyfhd_config["obs_checkpoint"] = False

            beam_checkpoint_file = Path(
                pyfhd_config["checkpoint_dir"], f"{checkpoint_name}_beam_checkpoint.h5"
            )
            if pyfhd_config["beam_checkpoint"] and not beam_checkpoint_file.exists():
                logger.warning(
                    "beam_checkpoint is set but beam checkpoint file does not "
                    "exist. Recalculating beam."
                )
                pyfhd_config["beam_checkpoint"] = False

            cal_checkpoint_file = Path(
                pyfhd_config["checkpoint_dir"],
                f"{checkpoint_name}_calibrate_checkpoint.h5",
            )
            if (
                pyfhd_config["calibrate_checkpoint"]
                and not cal_checkpoint_file.exists()
            ):
                logger.warning(
                    "calibrate_checkpoint is set but cal checkpoint file does "
                    "not exist. Recalculating cal."
                )
                pyfhd_config["calibrate_checkpoint"] = False

            grid_checkpoint_file = Path(
                pyfhd_config["checkpoint_dir"],
                f"{checkpoint_name}_gridding_checkpoint.h5",
            )
            if (
                pyfhd_config["gridding_checkpoint"]
                and not grid_checkpoint_file.exists()
            ):
                logger.warning(
                    "gridding_checkpoint is set but grid checkpoint file does "
                    "not exist. Recalculating grid."
                )
                pyfhd_config["gridding_checkpoint"] = False
        else:
            pyfhd_config["obs_checkpoint"] = False
            pyfhd_config["beam_checkpoint"] = False
            pyfhd_config["calibrate_checkpoint"] = False
            pyfhd_config["gridding_checkpoint"] = False

        if (
            not pyfhd_config["obs_checkpoint"]
            and not pyfhd_config["calibrate_checkpoint"]
        ):
            header_start = time.time()
            # Get the header
            pyfhd_header, params_data, antenna_header, antenna_data = extract_header(
                pyfhd_config
            )
            header_end = time.time()
            _print_time_diff(header_start, header_end, "pyfhd Header Created")

            params_start = time.time()
            # Get params
            params = create_params(pyfhd_header, params_data)
            params_end = time.time()
            _print_time_diff(params_start, params_end, "Params Created")

            visibility_start = time.time()
            vis_arr, vis_weights = extract_visibilities(
                pyfhd_header, params_data, pyfhd_config
            )
            visibility_end = time.time()
            _print_time_diff(visibility_start, visibility_end, "Visibilities Extracted")

            # Save the raw visibilities and weights if the option is set
            pyfhd_config["visibilities_path"] = Path(
                pyfhd_config["output_dir"], "visibilities"
            )
            if pyfhd_config["save_visibilities"]:
                pyfhd_config["visibilities_path"].mkdir(exist_ok=True)
                raw_vis_arr_path = Path(
                    pyfhd_config["visibilities_path"],
                    f"{pyfhd_config['obs_id']}_raw_vis_arr.h5",
                )
                save(raw_vis_arr_path, vis_arr, "visibilities")

            if pyfhd_config["save_weights"]:
                pyfhd_config["visibilities_path"].mkdir(exist_ok=True)
                weights_path = Path(
                    pyfhd_config["visibilities_path"],
                    f"{pyfhd_config['obs_id']}_raw_vis_weights.h5",
                )
                save(weights_path, vis_weights, "weights")

            # If you wish to reorder your visibilities, insert your function to
            # do that here.
            # If you wish to average your fits data by time or frequency, insert
            # your functions to do that here

            layout_start = time.time()
            layout = create_layout(antenna_header, antenna_data, pyfhd_config)
            layout_end = time.time()
            _print_time_diff(layout_start, layout_end, "Layout Dictionary Extracted")

            # Get obs
            obs_start = time.time()
            obs = create_obs(pyfhd_header, params, layout, pyfhd_config)
            obs_end = time.time()
            _print_time_diff(obs_start, obs_end, "Obs Dictionary Created")

            # If you decide to use the pyfhd checkpoint system, save the
            # uncalibrated visibility observation data and metadata now
            if pyfhd_config["save_checkpoints"]:
                checkpoint = {
                    "obs": obs,
                    "params": params,
                    "vis_arr": vis_arr,
                    "vis_weights": vis_weights,
                }
                save(obs_checkpoint_file, checkpoint, "obs_checkpoint")
                del checkpoint
                logger.info(
                    "Checkpoint Saved: Uncalibrated visibility parameters, "
                    "array and weights and the observation metadata dictionary "
                    f"saved into {obs_checkpoint_file}"
                )
        elif not pyfhd_config["calibrate_checkpoint"]:
            # if the cal checkpoint doesn't exist, load in the obs checkpoint
            # Load the checkpoint and initialize the required variables
            if pyfhd_config["obs_checkpoint"]:
                obs_checkpoint = load(obs_checkpoint_file)
                obs = obs_checkpoint["obs"]
                params = obs_checkpoint["params"]
                vis_arr = obs_checkpoint["vis_arr"]
                vis_weights = obs_checkpoint["vis_weights"]
                del obs_checkpoint
                logger.info(
                    "Checkpoint Loaded: Uncalibrated visibility parameters, "
                    "array and weights and the observation metadata dictionary "
                    f"loaded from {obs_checkpoint_file}"
                )

        # Give cal and vis_model_arr a value to avoid no value error
        # (Default = None, but can be overwritten below)
        cal = None
        vis_model_arr = None
        # If the calibration checkpoint exists, load it now before loading in the beam
        # to get the observation metadata and visibility parameters
        if pyfhd_config["calibrate_checkpoint"]:
            cal_checkpoint = load(cal_checkpoint_file)
            obs = cal_checkpoint["obs"]
            params = cal_checkpoint["params"]
            vis_arr = cal_checkpoint["vis_arr"]
            vis_model_arr = cal_checkpoint["vis_model_arr"]
            vis_weights = cal_checkpoint["vis_weights"]
            cal = cal_checkpoint["cal"]
            del cal_checkpoint
            logger.info(
                "Checkpoint Loaded: Calibrated and Flagged visibility parameters, "
                "array and weights, the flagged observation metadata dictionary "
                f"and the calibration dictionary loaded from {cal_checkpoint_file}"
            )

        if not pyfhd_config["beam_checkpoint"]:
            # Read in the beam from a file returning a psf dictionary
            psf_start = time.time()
            psf, antenna = create_psf(obs, pyfhd_config)
            psf_end = time.time()
            _print_time_diff(psf_start, psf_end, "Beam and PSF setup")

            # If you decide to use the pyfhd checkpoint system, save the beam
            # and antenna dicts now
            if pyfhd_config["save_checkpoints"]:
                checkpoint = {"psf": psf, "antenna": antenna}
                save(beam_checkpoint_file, checkpoint, "beam_checkpoint")
                del checkpoint
                logger.info(
                    "Checkpoint Saved: psf and antenana dictionaries saved into "
                    f"{beam_checkpoint_file}"
                )
        else:
            beam_checkpoint = load(beam_checkpoint_file)
            psf = beam_checkpoint["psf"]
            antenna = beam_checkpoint["antenna"]

            del beam_checkpoint
            logger.info(
                "Checkpoint Loaded: psf and antenana dictionaries loaded from "
                f"{beam_checkpoint_file}"
            )

        # Check if the calibrate checkpoint has been used, if not run the
        # calibration steps
        if (
            not pyfhd_config["calibrate_checkpoint"]
            and not pyfhd_config["gridding_checkpoint"]
        ):
            if pyfhd_config["deproject_w_term"] is not None:
                w_term_start = time.time()
                vis_arr = simple_deproject_w_term(
                    obs, params, vis_arr, pyfhd_config["deproject_w_term"]
                )
                w_term_end = time.time()
                _print_time_diff(
                    w_term_start, w_term_end, "Simple W-Term Deprojection Applied"
                )

            # Peform basic flagging
            if pyfhd_config["flag_basic"]:
                basic_flag_start = time.time()
                vis_weights, obs = vis_flag_basic(
                    vis_weights, vis_arr, obs, pyfhd_config
                )
                basic_flag_end = time.time()
                _print_time_diff(
                    basic_flag_start, basic_flag_end, "Basic Flagging Completed"
                )

            # Update the visibility weights
            weight_start = time.time()
            vis_weights, obs = vis_weights_update(vis_weights, obs, psf, params)
            weight_end = time.time()
            _print_time_diff(
                weight_start,
                weight_end,
                "Visibilities Weights Updated After Basic Flagging",
            )

            if pyfhd_config["model_file_path"] is not None:
                # Get the vis_model_arr from a UVFITS file or SAV files and flag
                # any issues
                vis_model_arr_start = time.time()
                vis_model_arr = vis_model_transfer(pyfhd_config, obs, params)
                vis_model_arr_end = time.time()
                _print_time_diff(
                    vis_model_arr_start,
                    vis_model_arr_end,
                    "Model Imported and Flagged From UVFITS",
                )

            # Skipped initializing the cal structure as it mostly just copies
            # values from the obs, params, config and the skymodel from FHD
            # However, there is resulting cal structure for logging and output
            # purposes to store the resulting gain and any other associated
            # arrays
            if pyfhd_config["calibrate_visibilities"]:
                logger.info("Beginning Calibration")
                cal_start = time.time()
                vis_arr, vis_model_arr, cal, obs, pyfhd_config = calibrate(
                    obs=obs,
                    psf=psf,
                    antenna=antenna,
                    params=params,
                    vis_arr=vis_arr,
                    vis_weights=vis_weights,
                    vis_model_arr=vis_model_arr,
                    pyfhd_config=pyfhd_config,
                )
                cal_end = time.time()
                _print_time_diff(
                    cal_start,
                    cal_end,
                    "Visibilities calibrated and cal dictionary with gains created",
                )

                if obs["n_pol"] >= 4:
                    qu_mixing_start = time.time()
                    cal["stokes_mix_phase"] = calibrate_qu_mixing(
                        vis_arr, vis_model_arr, vis_weights, obs
                    )
                    qu_mixing_end = time.time()
                    _print_time_diff(
                        qu_mixing_start,
                        qu_mixing_end,
                        'Calibrate QU-Mixing has finished, result in "'
                        '"cal["stokes_mix_phase"]',
                    )

                weight_start = time.time()
                vis_weights, obs = vis_weights_update(vis_weights, obs, psf, params)
                weight_end = time.time()
                _print_time_diff(
                    weight_start,
                    weight_end,
                    "Visibilities Weights Updated After Calibration",
                )

                if pyfhd_config["flag_visibilities"]:
                    flag_start = time.time()
                    vis_weights, obs = vis_flag(vis_arr, vis_weights, obs, params)
                    flag_end = time.time()
                    _print_time_diff(flag_start, flag_end, "Visibilities Flagged")
                    if np.max(vis_weights) == 0:
                        raise ValueError(
                            "All visibilities were flagged during the flagging "
                            "step, exiting pyfhd."
                        )

                noise_start = time.time()
                obs["vis_noise"] = vis_noise_calc(obs, vis_arr, vis_weights)
                noise_end = time.time()
                _print_time_diff(
                    noise_start, noise_end, "Noise Calculated and added to obs"
                )

                if pyfhd_config["save_checkpoints"]:
                    checkpoint = {
                        "obs": obs,
                        "params": params,
                        "vis_arr": vis_arr,
                        "vis_model_arr": vis_model_arr,
                        "vis_weights": vis_weights,
                        "cal": cal,
                    }
                    save(cal_checkpoint_file, checkpoint, "calibrate_checkpoint")
                    del checkpoint
                    logger.info(
                        "Checkpoint Saved: Calibrated and Flagged visibility "
                        "parameters, array and weights, the flagged observation "
                        "metadata dictionary and the calibration dictionary saved "
                        f"into {cal_checkpoint_file}"
                    )

        if pyfhd_config["cal_stop"]:
            logger.info(
                "The cal_stop option was used, calibration was finished, saving "
                "calibration files then exiting pyfhd"
            )
            pyfhd_config["metadata_dir"] = Path(pyfhd_config["output_dir"], "metadata")
            pyfhd_config["visibilities_path"] = Path(
                pyfhd_config["output_dir"], "visibilities"
            )
            pyfhd_config["metadata_dir"].mkdir(exist_ok=True)
            pyfhd_config["visibilities_path"].mkdir(exist_ok=True)

            if pyfhd_config["save_obs"]:
                obs_path = Path(
                    pyfhd_config["metadata_dir"], f"{pyfhd_config['obs_id']}_obs.h5"
                )
                logger.info(f"Saving the obs dictionary to {obs_path}")
                save(obs_path, obs, "obs")

            if pyfhd_config["save_params"]:
                params_path = Path(
                    pyfhd_config["metadata_dir"], f"{pyfhd_config['obs_id']}_params.h5"
                )
                logger.info(f"Saving params dictionary to {params_path}")
                save(params_path, params, "params")

            if pyfhd_config["save_cal"] and pyfhd_config["calibrate_visibilities"]:
                cal_path = Path(pyfhd_config["output_dir"], "calibration")
                cal_path.mkdir(exist_ok=True)
                cal_path = Path(cal_path, f"{pyfhd_config['obs_id']}_cal.h5")
                logger.info(f"Saving the calibration dictionary to {cal_path}")
                save(cal_path, cal, "cal")

            if pyfhd_config["save_weights"]:
                weights_path = Path(
                    pyfhd_config["visibilities_path"],
                    f"{pyfhd_config['obs_id']}_calibrated_vis_weights.h5",
                )
                logger.info(f"Saving the calibrated weights to {weights_path}")
                save(weights_path, vis_weights, "weights")

            if pyfhd_config["save_visibilities"]:
                cal_vis_arr_path = Path(
                    pyfhd_config["visibilities_path"],
                    f"{pyfhd_config['obs_id']}_calibrated_vis_arr.h5",
                )
                logger.info(f"Saving the calibrated visibilities to {cal_vis_arr_path}")
                save(cal_vis_arr_path, vis_arr, "visibilities")
            logger.info(
                "The cal_stop option was used, calibration was finished, exiting pyfhd"
            )
            pyfhd_successful = True
            _finish_pyfhd(pyfhd_start, psf, pyfhd_config)
            sys.exit(0)

        if "image_info" not in psf or (
            psf["image_info"]["image_power_beam_arr"] is not None
            and psf["image_info"]["image_power_beam_arr"].shape == 1
        ):
            # Turn off beam_per_baseline if image_power_beam_arr is
            # only one value
            pyfhd_config["beam_per_baseline"] = False

        if pyfhd_config["recalculate_grid"] or not pyfhd_config["gridding_checkpoint"]:
            grid_start = time.time()
            image_uv = np.empty(
                (obs["n_pol"], obs["elements"], obs["dimension"]), dtype=np.complex128
            )
            weights_uv = np.empty(
                (obs["n_pol"], obs["elements"], obs["dimension"]), dtype=np.complex128
            )
            variance_uv = np.empty((obs["n_pol"], obs["elements"], obs["dimension"]))
            uniform_filter_uv = np.empty((obs["elements"], obs["dimension"]))
            if vis_model_arr is not None:
                model_uv = np.empty(
                    (obs["n_pol"], obs["elements"], obs["dimension"]),
                    dtype=np.complex128,
                )
            else:
                model_uv = None

            for pol_i in range(obs["n_pol"]):
                logger.info(
                    f"Gridding has begun for polarization {obs['pol_names'][pol_i]}"
                )
                if pol_i == 0:
                    calculate_uniform_filter = True
                else:
                    # The filter is the same across pols, so only needs to be
                    # calculated once.
                    calculate_uniform_filter = False
                if pol_i > 1:
                    no_conjugate = True
                else:
                    no_conjugate = False
                if vis_model_arr is None:
                    vis_model_arr_use = None
                else:
                    vis_model_arr_use = vis_model_arr[pol_i]
                gridding_dict = visibility_grid(
                    vis_arr[pol_i],
                    vis_weights[pol_i],
                    obs,
                    psf,
                    params,
                    pol_i,
                    pyfhd_config,
                    calculate_uniform_filter=calculate_uniform_filter,
                    no_conjugate=no_conjugate,
                    model=vis_model_arr_use,
                )
                if len(gridding_dict.keys()) != 0:
                    image_uv[pol_i] = gridding_dict["image_uv"]
                    weights_uv[pol_i] = gridding_dict["weights"]
                    variance_uv[pol_i] = gridding_dict["variance"]
                    if calculate_uniform_filter:
                        uniform_filter_uv = gridding_dict["uniform_filter"]
                    obs["nf_vis"] = gridding_dict["obs"]["nf_vis"]
                    if vis_model_arr is not None:
                        model_uv[pol_i] = gridding_dict["model_return"]
                    logger.info(
                        "Gridding has finished for polarization "
                        f"{obs['pol_names'][pol_i]}"
                    )
                else:
                    logger.error("All data was flagged during gridding, exiting")
                    sys.exit(1)
            if obs["n_pol"] == 4:
                logger.info("Performing Crosspol reformatting")
                image_uv = crosspol_reformat(image_uv)
                weights_uv = crosspol_reformat(weights_uv)
                if vis_model_arr is not None:
                    model_uv = crosspol_reformat(model_uv)
            if pyfhd_config["gridding_plots"]:
                # TODO: move this after the checkpointing so an error in plotting
                # doesn't require rerunning gridding.
                logger.info(
                    "Plotting the continuum gridding outputs into "
                    f"{pyfhd_config['output_dir'] / 'plots' / 'gridding'}"
                )
                plot_gridding(
                    obs,
                    image_uv,
                    weights_uv,
                    variance_uv,
                    pyfhd_config,
                    model_uv=model_uv,
                    log=pyfhd_config["log_plots"],
                    sigma_clip_level=pyfhd_config["sigma_clipping"],
                    percentile_clip_level=pyfhd_config["percentile_clipping"],
                )
            if pyfhd_config["save_checkpoints"]:
                checkpoint = {
                    "image_uv": image_uv,
                    "weights_uv": weights_uv,
                    "variance_uv": variance_uv,
                    "uniform_filter_uv": uniform_filter_uv,
                }
                if vis_model_arr is not None:
                    checkpoint["model_uv"] = model_uv
                save(grid_checkpoint_file, checkpoint, "gridding_checkpoint")
                del checkpoint
                logger.info(
                    "Checkpoint Saved: The Gridded UV Planes saved into "
                    f"{grid_checkpoint_file}"
                )
            grid_end = time.time()
            _print_time_diff(grid_start, grid_end, "Visibilities gridded")
        else:
            grid_checkpoint = load(grid_checkpoint_file)
            image_uv = grid_checkpoint["image_uv"]
            weights_uv = grid_checkpoint["weights_uv"]
            variance_uv = grid_checkpoint["variance_uv"]
            uniform_filter_uv = grid_checkpoint["uniform_filter_uv"]
            if "model_uv" in grid_checkpoint:
                model_uv = grid_checkpoint["model_uv"]
            else:
                model_uv = None
            del grid_checkpoint
            logger.info(
                "Checkpoint Loaded: The Gridded UV Planes loaded from "
                f"{grid_checkpoint_file}"
            )

        # Call quickview to save the all the variables if set in the config.
        # Also create dirty images and save FITS files with the dirty images on
        # a per polarization basis
        if pyfhd_config["export_images"]:
            quickview(
                obs,
                psf,
                params,
                cal,
                vis_arr,
                vis_weights,
                image_uv,
                weights_uv,
                variance_uv,
                uniform_filter_uv,
                model_uv,
                pyfhd_config,
            )

        # Create the healpix HDF5 cubes and save them to disk
        if pyfhd_config["snapshot_healpix_export"]:
            healpix_snapshot_cube_generate(
                obs, psf, cal, params, vis_arr, vis_model_arr, vis_weights, pyfhd_config
            )
        pyfhd_successful = True
        _finish_pyfhd(pyfhd_start, psf, pyfhd_config)
    except Exception as e:
        logger.exception(
            f"An error occurred in pyfhd: {e}\n\tExiting pyfhd.", exc_info=True
        )
        pyfhd_successful = False
    finally:
        if not pyfhd_successful:
            pyfhd_end = time.time()
            runtime = timedelta(seconds=pyfhd_end - pyfhd_start)
            # Close all open h5 files
            if "psf" in locals() and isinstance(psf, h5py.File):
                psf.close()
            logger.info(
                f"pyfhd Run Unsuccessful for {pyfhd_config['obs_id']}\nTotal "
                f"Runtime (Days:Hours:Minutes:Seconds.Millseconds): {runtime}"
            )
            sys.exit(1)


def main():
    pyfhd_start = time.time()
    configargparser = pyfhd_parser()
    options = configargparser.parse_args()

    pyfhd_config = vars(options)

    # Get the time, used for various file names setup the name of the output directory
    run_time = time.localtime()

    pyfhd_config, output_dir_exists = setup_directory(pyfhd_config, run_time)

    # create the Logger
    with pyfhd_logger(pyfhd_config):
        # validate options
        pyfhd_config = pyfhd_setup(pyfhd_config, run_time, output_dir_exists)

        run_pyfhd(pyfhd_config, pyfhd_start)


if __name__ == "__main__":
    main()

import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

from astropy import constants
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
from .io.pyfhd_io import SAVEFILES, checkpoint_filenames, product_file, save, load
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
    if not pyfhd_config["get_sample_data"]:
        # Write a final collated yaml for the final pyfhd_config
        write_collated_yaml_config(
            pyfhd_config, Path(pyfhd_config["output_dir"], "config"), "-final"
        )
        # Save the config in a HDF5 file for ease of reading in previous
        # parameters from previous runs
        save(product_file("config", pyfhd_config), pyfhd_config, "pyfhd_config")
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
        product_filenames = checkpoint_filenames("setup", pyfhd_config, mkdir=True)

        if pyfhd_config["recalculate_all"]:
            header_start = time.time()
            # Get the header
            uvfits_path = Path(
                pyfhd_config["input_path"], pyfhd_config["obs_id"] + ".uvfits"
            )
            pyfhd_header, params_data, antenna_header, antenna_data = extract_header(
                uvfits_path
            )
            header_end = time.time()
            _print_time_diff(header_start, header_end, "pyfhd Header Created")

            params_start = time.time()
            # Get params
            params = create_params(pyfhd_header, params_data)

            save(product_filenames["params"], params, "params")
            params_end = time.time()
            _print_time_diff(params_start, params_end, "Params Created & Saved")

            layout_start = time.time()
            layout = create_layout(antenna_header, antenna_data, pyfhd_config)
            layout_end = time.time()
            _print_time_diff(layout_start, layout_end, "Layout Dictionary Extracted")

            # Get obs
            obs_start = time.time()
            obs = create_obs(pyfhd_header, params, layout, pyfhd_config)
            obs_end = time.time()

            save(product_filenames["obs"], obs, "obs")

            _print_time_diff(obs_start, obs_end, "Obs Dictionary Created")
        else:
            # Load in the setup outputs that we need
            obs = load(product_filenames["obs"])
            params = load(product_filenames["params"])

        # load raw (uncalibrated) vis & weights if needed
        if pyfhd_config["need_raw_vis"]:
            uvfits_path = Path(
                pyfhd_config["input_path"], pyfhd_config["obs_id"] + ".uvfits"
            )

            visibility_start = time.time()
            vis_arr, vis_weights = extract_visibilities(
                uvfits_path=uvfits_path, n_pol=obs["n_pol"]
            )
            visibility_end = time.time()
            _print_time_diff(visibility_start, visibility_end, "Visibilities Extracted")

            # If you wish to reorder your visibilities, insert your function to
            # do that here.
            # If you wish to average your fits data by time or frequency, insert
            # your functions to do that here

            logger.info("Read in Uncalibrated visibilities and weights.")

        # Figure out psf_dim if it is not set. This is needed beyond the psf
        # structure, so doing it here where we have the required info
        if pyfhd_config["psf_dim"] is None:
            # set psf_dim from antenna size
            psf_dim = np.ceil(
                pyfhd_config["antenna_size"]
                * 2
                * np.max(obs["baseline_info"]["freq"])
                / (constants.c.value * obs["kpix"])
            )
            pyfhd_config["psf_dim"] = int(np.ceil(psf_dim / 2) * 2)

        # setup/load the beam if needed.
        if pyfhd_config["need_beam"]:
            product_filenames.update(
                checkpoint_filenames("beam", pyfhd_config, mkdir=True)
            )

            if pyfhd_config["recalculate_beam"]:
                # Read in the beam from a file returning a psf dictionary
                psf_start = time.time()
                psf, antenna = create_psf(obs, pyfhd_config)
                psf_end = time.time()
                _print_time_diff(psf_start, psf_end, "Beam and PSF setup")

                if pyfhd_config["save_beam"]:
                    # N.B.: psf was already saved in create_psf
                    save(product_filenames["antenna"], antenna, "antenna")
            else:
                # Load in the beam products that we need
                antenna = load(product_filenames["antenna"])
                psf = load(
                    product_filenames["psf"], lazy_load=pyfhd_config["lazy_load_beam"]
                )

        # Take care of some things on raw vis here if needed
        if pyfhd_config["need_raw_vis"]:
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
            vis_weights, obs = vis_weights_update(
                vis_weights, obs=obs, psf_dim=pyfhd_config["psf_dim"], params=params
            )
            weight_end = time.time()
            _print_time_diff(
                weight_start,
                weight_end,
                "Visibilities Weights Updated After Basic Flagging",
            )
            # update the saved obs
            save(product_filenames["obs"], obs, "obs")

        # load in the model visibilities if needed
        if pyfhd_config["need_model_vis"]:
            if pyfhd_config["cal_model_file_path"] is not None:
                # Get the vis_model_arr from a UVFITS file or SAV files and flag
                # any issues
                vis_model_arr_start = time.time()
                vis_model_arr = vis_model_transfer(pyfhd_config, obs, params)
                vis_model_arr_end = time.time()
                _print_time_diff(
                    vis_model_arr_start,
                    vis_model_arr_end,
                    "Model Visibilities Imported and Flagged",
                )
            elif not pyfhd_config["recalculate_cal_model_vis"]:
                # If we don't need to recalculate model vis, just load them.
                # If they exist they won't be recalculated in calibration.
                # If they need to be recalculated that will be done in calibration.
                filename = product_file("cal_model_vis_arr", pyfhd_config)
                filename.parent.mkdir(exist_ok=True)
                product_filenames["cal_model_vis_arr"] = filename

                # Load in the beam products that we need
                vis_model_arr = load(product_filenames["cal_model_vis_arr"])
            else:
                vis_model_arr = None

        product_filenames.update(checkpoint_filenames("cal", pyfhd_config, mkdir=True))

        if pyfhd_config["recalculate_cal"]:
            logger.info("Beginning Calibration")
            cal_start = time.time()
            # This cal structure is smaller than in FHD because we avoid
            # duplicating values from obs, params and config into cal structure.
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
            vis_weights, obs = vis_weights_update(
                vis_weights, obs=obs, psf_dim=psf["dim"], params=params
            )
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

            # update the saved obs
            save(product_filenames["obs"], obs, "obs")
            # save the calibration products
            save(product_filenames["cal"], cal, "cal")

            if pyfhd_config["save_visibilities"]:
                # note: model vis arr is saved in calibrate (if calculated)
                save(product_filenames["cal_vis_arr"], vis_arr, "visibilities")
                save(product_filenames["cal_vis_weights"], vis_weights, "weights")

        else:
            if pyfhd_config["need_cal_vis"]:
                vis_arr = load(product_filenames["cal_vis_arr"])
                vis_weights = load(product_filenames["cal_vis_weights"])

        if pyfhd_config["cal_stop"]:
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
            # TODO: this is updating the pyfhd config. Should it warn?
            # These keys in psf can only exist for beams transferred in from FHD
            # at the moment. Maybe the docs should reflect that?
            pyfhd_config["beam_per_baseline"] = False

        product_filenames.update(
            checkpoint_filenames("gridding", pyfhd_config, mkdir=True)
        )
        if pyfhd_config["model_exists"]:
            product_filenames["grid_model_uv"] = product_file(
                "grid_model_uv", pyfhd_config, mkdir=True
            )
        if pyfhd_config["recalculate_grid"]:
            grid_start = time.time()
            image_uv = np.empty(
                (obs["n_pol"], obs["elements"], obs["dimension"]), dtype=np.complex128
            )
            weights_uv = np.empty(
                (obs["n_pol"], obs["elements"], obs["dimension"]), dtype=np.complex128
            )
            variance_uv = np.empty((obs["n_pol"], obs["elements"], obs["dimension"]))
            uniform_filter_uv = np.empty((obs["elements"], obs["dimension"]))
            if pyfhd_config["model_exists"]:
                model_uv = np.empty(
                    (obs["n_pol"], obs["elements"], obs["dimension"]),
                    dtype=np.complex128,
                )

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
                if pyfhd_config["model_exists"]:
                    vis_model_arr_use = vis_model_arr[pol_i]
                else:
                    vis_model_arr_use = None

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
                    if pyfhd_config["model_exists"]:
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
                if pyfhd_config["model_exists"]:
                    model_uv = crosspol_reformat(model_uv)

            # update the saved obs
            save(product_filenames["obs"], obs, "obs")

            # save gridding products
            save(product_filenames["grid_data_uv"], image_uv, "image_uv")
            save(product_filenames["grid_weights_uv"], weights_uv, "weights_uv")
            save(product_filenames["grid_variance_uv"], variance_uv, "variance_uv")
            save(
                product_filenames["uniform_filter_uv"],
                uniform_filter_uv,
                "uniform_filter_uv",
            )
            if pyfhd_config["model_exists"]:
                save(product_filenames["grid_model_uv"], model_uv, "model_uv")

            grid_end = time.time()
            _print_time_diff(grid_start, grid_end, "Visibilities gridded")

        elif pyfhd_config["gridding_plots"] or pyfhd_config["export_images"]:
            image_uv = load(product_filenames["grid_data_uv"])
            weights_uv = load(product_filenames["grid_weights_uv"])
            variance_uv = load(product_filenames["grid_variance_uv"])
            uniform_filter_uv = load(product_filenames["uniform_filter_uv"])
            if pyfhd_config["model_exists"]:
                model_uv = load(product_filenames["grid_model_uv"])

            logger.info("Checkpoint Loaded: The Gridded UV Planes loaded.")

        if pyfhd_config["gridding_plots"]:
            logger.info(
                "Plotting the continuum gridding outputs into "
                f"{pyfhd_config['output_dir'] / 'plots' / 'gridding'}"
            )
            if pyfhd_config["model_exists"]:
                model_uv_use = model_uv
            else:
                model_uv_use = None

            plot_gridding(
                obs,
                image_uv,
                weights_uv,
                variance_uv,
                pyfhd_config,
                model_uv=model_uv_use,
                log=pyfhd_config["log_plots"],
                sigma_clip_level=pyfhd_config["sigma_clipping"],
                percentile_clip_level=pyfhd_config["percentile_clipping"],
            )

        # Call quickview to save the all the variables if set in the config.
        # Also create dirty images and save FITS files with the dirty images on
        # a per polarization basis
        if pyfhd_config["export_images"]:
            if pyfhd_config["model_exists"]:
                model_uv_use = model_uv
            else:
                model_uv_use = None
            quickview(
                obs=obs,
                psf=psf,
                image_uv=image_uv,
                weights_uv=weights_uv,
                uniform_filter_uv=uniform_filter_uv,
                model_uv=model_uv_use,
                pyfhd_config=pyfhd_config,
            )

        if (
            pyfhd_config["snapshot_healpix_export"]
            and not pyfhd_config["recalculate_healpix"]
        ):
            # Now we have the info we need to figure out what all the expected healpix
            # cube files are so we can actually test if they are all present
            if pyfhd_config["split_ps_export"]:
                cube_name = ["hpx_even", "hpx_odd"]
            else:
                cube_name = ["healpix_cube"]
            expected_files = []
            for cube in cube_name:
                for pol_i in range(obs["n_pol"]):
                    expected_files.append(
                        f"{pyfhd_config['obs_id']}_{cube}_{obs['pol_names'][pol_i]}.h5"
                    )
            healpix_folder = Path(
                pyfhd_config["output_dir"], SAVEFILES["healpix_cube"]["folder"]
            )
            files_exist = []
            for filename in expected_files:
                filepath = healpix_folder / filename
                files_exist.append(filepath.exists())

            if not np.all(files_exist):
                logger.warning(
                    "recalculate_healpix not set but healpix files do "
                    "not all exist and are needed. Recalculating healpix."
                )
                pyfhd_config["recalculate_healpix"] = True

        # Create the healpix HDF5 cubes and save them to disk
        if pyfhd_config["recalculate_healpix"]:
            healpix_snapshot_cube_generate(
                obs=obs,
                psf=psf,
                params=params,
                vis_arr=vis_arr,
                vis_model_arr=vis_model_arr,
                vis_weights=vis_weights,
                pyfhd_config=pyfhd_config,
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

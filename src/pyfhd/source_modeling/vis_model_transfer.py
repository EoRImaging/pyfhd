import importlib_resources
import logging
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.io import readsav

from pyfhd.data_setup.obs import create_obs
from pyfhd.data_setup.uvfits import create_params, create_layout
from pyfhd.data_setup.uvfits import extract_visibilities, extract_header
from pyfhd.io.pyfhd_io import recarray_to_dict, save, load
from pyfhd.pyfhd_tools.pyfhd_utils import run_command


def vis_model_transfer(
    pyfhd_config: dict, obs: dict, params: dict, logger: logging.Logger
) -> tuple[NDArray[np.complex128], dict]:
    """
    Transfer in a simulated model of the visibilities from either a sav file or uvfits file.

    Parameters
    ----------
    pyfhd_config : dict
        pyfhd's configuration dictionary containing all the options for a pyfhd run
    obs : dict
        The Observation Metadata dictionary
    params : dict
        Visibility metadata dictionary
    logger : logging.Logger
        pyfhd's logger

    Returns
    -------
    vis_model_arr : NDArray[np.complex128]
        Simulated model for the visibilities

    See Also
    --------
    pyfhd.source_modeling.vis_model_transfer.import_vis_model_from_sav : Import model from a sav file
    pyfhd.source_modeling.vis_model_transfer.import_vis_model_from_uvfits : Import model from a uvfits file
    """
    if pyfhd_config["model_file_type"] == "sav":
        vis_model, params_model, obs_model = import_vis_model_from_sav(
            pyfhd_config, obs, logger
        )
    elif pyfhd_config["model_file_type"] == "uvfits":
        vis_model, params_model, obs_model = import_vis_model_from_uvfits(
            pyfhd_config, obs, logger
        )
    elif pyfhd_config["model_file_type"] == "h5":
        # Assume it's a pyfhd h5 file
        model = load(pyfhd_config["model_file_path"], logger=logger)
        vis_model = model["vis_model_arr"]
        params_model = model["params"]
        obs_model = model["obs"]
    else:
        logger.error("You chose a file type pyfhd can't import, exiting")
        raise ValueError(
            f"File type {pyfhd_config['model_file_type']} not supported. Please use 'sav' or 'uvfits'."
        )

    if pyfhd_config["flag_model"]:
        vis_model = flag_model_visibilities(
            vis_model, params_data=params, params_model=params_model, obs_data=obs, 
            obs_model=obs_model, pyfhd_config=pyfhd_config, logger=logger
        )
    else:
        logger.warning(
            "You have chosen not to flag the model visibilities, so pyfhd will not account for difference in time steps between the data and the model "
            "or any flagged tiles. This may lead to incorrect calibration results if the model visibilities are not compatible with the data visibilities."
        )

    if pyfhd_config["save_model"]:
        model_dir = Path(pyfhd_config["output_dir"], "model")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = Path(model_dir, f"{pyfhd_config['obs_id']}_vis_model.h5")
        model = {
            "vis_model_arr": vis_model,
            "params": params_model,
        }
        save(model_file, model, "vis_model", logger=logger)

    return vis_model


def import_vis_model_from_sav(
    pyfhd_config: dict, obs: dict, logger: logging.Logger
) -> tuple[NDArray[np.complex128], dict]:
    """
    Read a model visibility array and metadata from multiple IDL sav files in a directory
    given by pyfhd_config['model_file_path'], or in a FHD-style directory structure given
    by pyfhd_config['model_file_path']. The data is assumed to be in the format of
    <obs_id>_params.sav, <obs_id>_obs.sav, and <obs_id>_vis_model_<pol_name>.sav. The
    pol_name follows the pol_names in the obs dictionary, ['XX','YY','XY','YX','I','Q',
    'U','V'].

    Parameters
    ----------
    pyfhd_config : dict
        The pyfhd config dictionary
    obs : dict
        The dictionary containing observation data and metadata
    logger : logging.Logger
        pyfhd's logger

    Returns
    -------
    vis_model_arr : NDArray[np.complex128]
        Simulated model visibilities
    params_model : dict
        The parameter dictionary of the input model
    obs_model : dict
        The observation dictionary of the input model
    """

    fhd_subdirs = {
        "params": "metadata",
        "obs": "metadata",
        "vis": "vis_data",
    }

    try:
        path = Path(
            pyfhd_config["model_file_path"], f"{pyfhd_config['obs_id']}_params.sav"
        )
        if not path.exists():
            path = Path(
                pyfhd_config["model_file_path"],
                fhd_subdirs["params"],
                f"{pyfhd_config['obs_id']}_params.sav",
            )
        params_model = readsav(path)
        params_model = recarray_to_dict(params_model.params)

        path = Path(
            pyfhd_config["model_file_path"], f"{pyfhd_config['obs_id']}_obs.sav"
        )
        if not path.exists():
            path = Path(
                pyfhd_config["model_file_path"],
                fhd_subdirs["obs"],
                f"{pyfhd_config['obs_id']}_obs.sav",
            )
        obs_model = readsav(path)
        obs_model = recarray_to_dict(obs_model.obs)

        # Read in the first polarization from pol_names
        pol_i = 0
        vis_path_parts = [pyfhd_config["model_file_path"]]
        path = Path(
            *vis_path_parts,
            f"{pyfhd_config['obs_id']}_vis_model_{obs['pol_names'][pol_i]}.sav",
        )
        if not path.exists():
            vis_path_parts = [pyfhd_config["model_file_path"], fhd_subdirs["vis"]]
            path = Path(
                *vis_path_parts,
                f"{pyfhd_config['obs_id']}_vis_model_{obs['pol_names'][pol_i]}.sav",
            )
        curr_vis_model = readsav(path)
        if isinstance(curr_vis_model, dict):
            curr_vis_model = curr_vis_model["vis_model_ptr"]
        elif isinstance(curr_vis_model, np.recarray):
            curr_vis_model = curr_vis_model.vis_model_ptr
        # Sometimes arrays a packed in via a pointer, if so extract it
        if curr_vis_model.size == 1:
            curr_vis_model = curr_vis_model[0]
        curr_vis_model = curr_vis_model.transpose().astype(np.complex128)
        # The shape should be n_pol, n_freq, n_time * n_baselines
        vis_model_shape = [obs["n_pol"]] + list(curr_vis_model.shape)
        vis_model_arr = np.empty(vis_model_shape, dtype=np.complex128)
        vis_model_arr[pol_i] = curr_vis_model
        for pol_i in range(1, obs["n_pol"]):
            path = Path(
                *vis_path_parts,
                f"{pyfhd_config['obs_id']}_vis_model_{obs['pol_names'][pol_i]}.sav",
            )
            curr_vis_model = readsav(path)
            if isinstance(curr_vis_model, dict):
                curr_vis_model = curr_vis_model["vis_model_ptr"]
            elif isinstance(curr_vis_model, np.recarray):
                # Should be a rec array containing one item vis_model_ptr
                curr_vis_model = curr_vis_model.vis_model_ptr
            # Sometimes arrays a packed in via a pointer, if so extract it
            if curr_vis_model.size == 1:
                curr_vis_model = curr_vis_model[0]
            vis_model_arr[pol_i] = curr_vis_model.transpose().astype(np.complex128)
    except FileNotFoundError as e:
        logger.error(
            f"pyfhd failed to load in the model visibilities and metadata from filepath: {path}"
        )
        exit()
    return vis_model_arr, params_model, obs_model


def import_vis_model_from_uvfits(
    pyfhd_config: dict, obs: dict, logger: logging.Logger
) -> tuple[NDArray[np.complex128], dict]:
    """Read a model visibility array in from a `uvfits` with filepath given
    by pyfhd_config['model_file_path']. Reads data in via
    `pyfhd.data_setup.uvfits import extract_visibilities`.

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `pyfhd.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    obs : dict
        The observation dictionary as populated by `pyfhd.data_setup.obs.create_obs`

    Returns
    -------
    vis_model_arr : NDArray[np.complex128]
        Simulated model for the visibilities
    params_model : dict
        The parameters for said model used for flagging
    """

    header_model, params_data_model, antenna_header, antenna_data = extract_header(
        pyfhd_config, logger, model_uvfits=True
    )

    params_model = create_params(header_model, params_data_model, logger)

    vis_model_arr, _ = extract_visibilities(
        header_model, params_data_model, pyfhd_config, logger
    )

    layout_model = create_layout(antenna_header, antenna_data, pyfhd_config, logger)

    obs_model = create_obs(
        header_model, params_model, layout_model, pyfhd_config, logger
    )

    return vis_model_arr, params_model, obs_model


class _FlaggingInfoCounter(object):
    """Something to count and hold numbers to do with baselines"""

    def __init__(self, params: dict, obs: dict):
        """
        Given a populated params and obs dict (as populated by
        `pyfhd.data_setup.uvfits.create_<name>`), calculate many useful quantities
        to do with antenna (tile) names and numbers, expected number of cross and
        auto correlations etc.
        """

        self.unique_times = np.unique(obs["baseline_info"]["jdate"])
        self.num_times = len(self.unique_times)

        ant_names1 = np.unique(params["antenna1"])
        ant_names2 = np.unique(params["antenna2"])

        self.num_visis = len(params["antenna1"])

        # If there are no auto-correlations, you don't get every unique tile
        # in either antenna1 or antenna2, so do a unique on both of them to be sure
        self.ant_names = np.unique(np.append(ant_names1, ant_names2))
        self.num_ants = len(self.ant_names)

        # indexes of the auto-correlations
        self.auto_locs = params["antenna1"] == params["antenna2"]
        self.num_autos = np.count_nonzero(self.auto_locs)

        if self.num_autos == 0:
            self.num_autos_per_time = 0
        else:
            self.num_autos_per_time = self.num_ants

        # indexes of the cross-correlations
        self.cross_locs = params["antenna1"] != params["antenna2"]

        # how many cross-correlations there should be per time step
        self.num_cross_per_time = int((self.num_ants * (self.num_ants - 1)) / 2)

        # number of visibilities per time step
        self.num_visi_per_time_step = self.num_cross_per_time + self.num_autos_per_time

        # within a single time step, where the cross-correlations are indexed
        # can use this while iterating over time to select the crosses only
        self.cross_locs_per_time = np.where(
            self.cross_locs[: self.num_visi_per_time_step]
        )[0]

        # within a single time step, where the cross-correlations are indexed
        # can use this while iterating over time to select the crosses only
        self.auto_locs_per_time = np.where(
            self.auto_locs[: self.num_visi_per_time_step]
        )[0]

        self.ant1_single_time = params["antenna1"][: self.num_visi_per_time_step]
        self.ant2_single_time = params["antenna2"][: self.num_visi_per_time_step]


def flag_model_visibilities(
    vis_model_arr: NDArray[np.complex128],
    *,
    params_data: dict,
    params_model: dict,
    obs_data: dict,
    obs_model: dict,
    pyfhd_config: dict,
    logger: logging.Logger,
) -> NDArray[np.complex128]:
    """
    Match the times and the tile flags between the data and the input model, and
    check that the uvfits are compatible. Needs to check if auto-correlations are present

    Parameters
    ----------
    vis_model_arr : NDArray[np.complex128]
        The visibility array from the intput model uvfits or sav file
    params_data : dict
        The params metadata from the observation uvfits
    params_model : dict
        The params metadata from the input model uvfits or sav file
    obs_data : dict
        The observaton dictionary containing metadata from the observation uvfits file
    obs_model : dict
        The observaton dictionary containing metadata from the input uvfits or sav file
    pyfhd_config : dict
        The pyfhd configuration dictionary
    logger : logging.Logger
        The pyfhd logger

    Returns
    -------
    vis_model_arr_flagged: NDArray[np.complex128]
        The flagged model visibility array
    """

    # Calculate a number of things we'll need to compare the data to the model
    flaginfo_data = _FlaggingInfoCounter(params_data, obs_data)
    flaginfo_model = _FlaggingInfoCounter(params_model, obs_model)

    # Calculate a tolerance for the time difference between the transferred
    # model and the data in Julian date. If the difference is greater than
    # this tolerance, then the model and data are not compatible. Given the
    # modelling software, we might expect small differences in JD calculations.
    # Calculate half of the time integration width in JD.
    time_tolerance = (obs_data["time_res"] / (24.0 * 60 * 60)) / 1e2
    time_half_res = (obs_data["time_res"] / 2.0) / (24.0 * 60 * 60)

    # The convention for Julian dates is to mark the center of the time step
    # (AIPS Memo compliant). We will check if the input model matches the
    # expected convention or if it marks the beginning of the time step.
    # We will also account for the case where the data is a subset of the
    # model.

    # Option 1: Model matches the standard Julian date convention
    matched_times_std = np.full(flaginfo_data.num_times, -1, dtype=int)

    # Option 2: Model's Julian date convention is the mark the beginning of
    #           the time step
    matched_times_beg = np.full(flaginfo_data.num_times, -1, dtype=int)

    for time_index, time in enumerate(flaginfo_data.unique_times):
        diffs_std = np.abs(flaginfo_model.unique_times - time)
        diffs_beg = np.abs(flaginfo_model.unique_times - (time - time_half_res))

        # Minimum value and its index
        min_ind = np.argmin(diffs_std)
        min_val = diffs_std[min_ind]

        # If JD difference is below the tolerance, then it is a match
        if min_val < time_tolerance:
            matched_times_std[time_index] = min_ind

        # Minimum value and its index
        min_ind = np.argmin(diffs_beg)
        min_val = diffs_beg[min_ind]

        # If JD difference is below the tolerance, then it is a match
        if min_val < time_tolerance:
            matched_times_beg[time_index] = min_ind

    # Count how many matches were successful (i.e. != -1)
    n_matched_std = np.count_nonzero(matched_times_std != -1)
    n_matched_beg = np.count_nonzero(matched_times_beg != -1)

    # If no times matched, then error.
    if n_matched_std == 0 and n_matched_beg == 0:
        data_path = str(
            Path(pyfhd_config["input_path"], pyfhd_config["obs_id"] + ".uvfits")
        )
        model_path = (
            f"{pyfhd_config['model_file_path']}.{pyfhd_config['model_file_type']}"
        )
        raise ValueError(
            f"Could not match the time steps in the data uvfits: {data_path}"
            f" and model uvfits in {model_path}."
        )

    # Choose the option with the most matched times
    if n_matched_std >= n_matched_beg:
        model_times_to_use = matched_times_std
    else:
        model_times_to_use = matched_times_beg

    # Now to flag the model - some models have no flagged tiles (antennas),
    # whereas the data might have flagged tiles (and so missing baselines).
    # This means we need to flag the missing tiles out of the data and
    # reshape the model

    # If less antennas in the model than the data, we can't calibrate the whole
    # dataset so just error for now
    if flaginfo_model.num_ants < flaginfo_data.num_ants:
        model_path = pyfhd_config["model_file_path"] + pyfhd_config["model_file_type"]
        raise ValueError(
            f"There are less antennas (tiles) in the model "
            f"{model_path} than in the data, so cannot calibrate the "
            "whole dataset. Please check the model "
            "and try again. Exiting now."
        )

    # Test to see if there are auto-correlations in data
    if flaginfo_model.num_autos == 0:
        logger.warning("There are no auto-correlations present in model.")

    # This is where the fun begins - pyuvdata uses the tile number as written
    # in the 'TILE' column in the metafits file to encode baselines (and so
    # populate params[antenna1] and params[antenna2]). The numbers are MWA
    # assigned, and have nothing to do with antenna index. Birli and WODEN use
    # the tile index (one-indexed as the BASELINE encoding is one-indexed).
    # So to match a flagged tile from the data to the model, need to work out
    # the index of any flagged tiles in the data. All tile names are read in
    # from the metafits

    flag_indexes = []

    # if the data just have tile indexes, the maximum should be the number
    # of tiles - use that to work out if we have Birli of pyuvdata input

    # we should have a pyuvdata input in this case as a tile name is greater
    # than the number of tiles
    if np.max(flaginfo_data.ant_names) > len(obs_data["baseline_info"]["tile_names"]):

        # Loop over all possible antenna (tile) names, and if they're not in the
        # list of antennas in this data set, append to flag_indexes
        for ant_ind, ant_name in enumerate(obs_data["baseline_info"]["tile_names"]):
            if ant_name not in flaginfo_data.ant_names:
                flag_indexes.append(ant_ind + 1)

    # tiles are named by their index (1 indexed as per uvfits standard)
    else:
        for ant_name in range(1, len(obs_data["baseline_info"]["tile_names"]) + 1):
            if ant_name not in flaginfo_data.ant_names:
                flag_indexes.append(ant_name)

    if len(flag_indexes) > 0:
        logger.info(
            f"Found flagged tiles {flag_indexes} in the data, flagging from the model"
        )

    # This gives us true/false if the visibilities should be included
    # for a single time step based on antenna1 and antenna2
    include_per_time_ant1 = np.isin(
        flaginfo_model.ant1_single_time, flag_indexes, invert=True
    )
    include_per_time_ant2 = np.isin(
        flaginfo_model.ant2_single_time, flag_indexes, invert=True
    )

    # Doing a logic union combines info from ant1 and ant2
    model_include_per_time = np.nonzero(include_per_time_ant1 & include_per_time_ant2)[
        0
    ]

    # Check if the model has auto-correlations
    if (flaginfo_model.num_autos) > 0:
        data_include_per_time = np.sort(
            np.concat(
                [flaginfo_data.cross_locs_per_time, flaginfo_data.auto_locs_per_time]
            )
        )
    else:
        # If no auto-correlations, just use the cross-correlations, Keep the autos as zeroes
        data_include_per_time = flaginfo_data.cross_locs_per_time
        if flaginfo_data.num_autos > 0:
            logger.warning(
                "The data has auto-correlations, but the model does not. "
                "Setting the auto correlations locations to zero in the model."
            )

    # empty holder for the flagged model - this should be the same shape
    vis_model_arr_flagged = np.zeros(
        (obs_data["n_pol"], obs_data["n_freq"], flaginfo_data.num_visis),
        dtype=np.complex128,
    )

    # For each time step that matches the data, copy across any visibilities
    # that aren't to be flagged
    for t_data_ind, t_model_ind in enumerate(model_times_to_use):

        # Subset of cross-corrs from flagged model to select for this time step
        t_flag_inds = (
            t_data_ind * flaginfo_data.num_visi_per_time_step + data_include_per_time
        )
        # Subset of cross-corrs from full model to select for this time step
        t_model_inds = (
            t_model_ind * flaginfo_model.num_visi_per_time_step + model_include_per_time
        )
        # Stick it in the flagged model
        vis_model_arr_flagged[:, :, t_flag_inds] = vis_model_arr[:, :, t_model_inds]

    return vis_model_arr_flagged


def convert_vis_model_arr_to_sav(
    vis_model_arr: NDArray[np.complex128],
    pyfhd_config: dict,
    logger: logging.Logger,
    model_vis_dir: str,
    n_pol: int,
):
    """
    Converts the contents of `vis_model_arr` into an FHD .sav file format
    so we can import into existing IDL code with ease. First writes data to
    `hdf5` format, then uses IDL code template to convert to IDL `.sav` format
    compatible with FHD. Sticks the outputs into `model_vis_dir`.

    Parameters
    ----------
    vis_model_arr : NDArray[np.complex128]
        Complex array hold the model visibilities
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `pyfhd.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    logger : logging.Logger
        pyfhd logger to feed information to
    model_vis_dir : str
        Directory location to write the output files to
    n_pol : int
        Number of polarisations to write out (each is written to an individual)
        `.sav` file
    """

    pol_names = ["XX", "YY", "XY", "YX"]

    logger.info(
        f"vis_model_transfer: saving {model_vis_dir}/{pyfhd_config['obs_id']}_vis_model.h5"
    )

    with h5py.File(f"{model_vis_dir}/{pyfhd_config['obs_id']}_vis_model.h5", "w") as hf:

        for pol, pol_name in enumerate(pol_names[:n_pol]):

            hf.create_dataset(
                f"{pyfhd_config['obs_id']}_vis_model_{pol_name}",
                data=vis_model_arr[pol].transpose(),
            )

        hf.close()

    # Grab the template IDL code and transfer so people can see what code was used
    # and modify if they want
    model_arr_convert_pro = importlib_resources.files("pyfhd.templates").joinpath(
        "convert_model_arr_to_sav.pro"
    )
    shutil.copy(model_arr_convert_pro, model_vis_dir)

    # Move into the output directory so IDL can see all the .pro files
    os.chdir(model_vis_dir)

    # Run the IDL code
    logger.info(
        f"vis_model_transfer: converting {model_vis_dir}/{pyfhd_config['obs_id']}_vis_model.h5 to .sav format"
    )

    idl_command = f"idl -IDL_DEVICE ps -e convert_model_arr_to_sav -args {model_vis_dir} {pyfhd_config['obs_id']} {n_pol}"

    run_command(idl_command, False)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from pathlib import Path

matplotlib.use("pdf")


def plot_cals(obs: dict, cal: dict, pyfhd_config: dict):
    """
    Plot the calibration solutions, the calibration residuals, and the raw calibration solutions
    in a grid of 128 per page.

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    cal : dict
        Calibration dictionary
    pyfhd_config : dict
        Run option dictionary
    """

    # Plotting only unflagged frequencies
    freq_i_use = np.where(obs["baseline_info"]["freq_use"])[0]
    freq_arr_use = obs["baseline_info"]["freq"][freq_i_use] / 1e6

    # Plotting only unflagged tiles
    tile_i_use = np.where(obs["baseline_info"]["tile_use"])[0]

    # Plotting amplitude and phase for the gain, gain_residual, and their sum (raw solutions)
    obs_id = pyfhd_config["obs_id"]
    cal_plot_dir = Path(pyfhd_config["output_dir"], "plots", "calibration")
    cal_plot_dir.mkdir(parents=True, exist_ok=True)
    n_types = 3  # plotting three types of amp and phase gains
    save_path_roots = [
        Path(cal_plot_dir, f"{obs_id}_cal_amp"),
        Path(cal_plot_dir, f"{obs_id}_cal_phase"),
        Path(cal_plot_dir, f"{obs_id}_cal_residual_amp"),
        Path(cal_plot_dir, f"{obs_id}_cal_residual_phase"),
        Path(cal_plot_dir, f"{obs_id}_cal_raw_amp"),
        Path(cal_plot_dir, f"{obs_id}_cal_raw_phase"),
    ]

    # Calibration solutions are referenced mutliple times, put them in variables for speed
    cal_sol_amp = np.abs(cal["gain"][:, freq_i_use, :])
    cal_sol_phase = np.arctan2(
        (cal["gain"][:, freq_i_use, :]).imag, (cal["gain"][:, freq_i_use, :]).real
    )
    cal_raw_amp = np.abs(
        cal["gain"][:, freq_i_use, :] + cal["gain_residual"][:, freq_i_use, :]
    )
    cal_raw_phase = np.arctan2(
        (cal["gain"][:, freq_i_use, :] + cal["gain_residual"][:, freq_i_use, :]).imag,
        (cal["gain"][:, freq_i_use, :] + cal["gain_residual"][:, freq_i_use, :]).real,
    )

    # NOTE: The residuals that are plotted are the differences in raw amp/phase and solution amp/phase,
    # *not* the amp/phase of the raw and solution difference
    cal_res_amp = cal_raw_amp - cal_sol_amp
    cal_res_phase = np.unwrap(cal_raw_phase, axis=1) - np.unwrap(cal_sol_phase, axis=1)

    # Find the min/max amplitude and phase for plotting, only using unflagged tiles
    amp_minmax = np.zeros((6))
    phase_minmax = np.zeros((6))
    amp_minmax[0:2] = [np.nanmin(cal_sol_amp), np.nanmax(cal_sol_amp)]
    amp_minmax[2:4] = [np.nanmin(cal_res_amp), np.nanmax(cal_res_amp)]
    amp_minmax[4:6] = [np.nanmin(cal_raw_amp), np.nanmax(cal_raw_amp)]
    phase_minmax[0:2] = [-np.pi - 0.1, np.pi + 0.1]
    phase_minmax[2:4] = [np.nanmin(cal_res_phase), np.nanmax(cal_res_phase)]
    phase_minmax[4:6] = [-np.pi - 0.1, np.pi + 0.1]

    # Create patches for the legend
    patches = [
        plt.plot(
            [],
            [],
            color=(["blue", "red"])[pol_i],
            linestyle="solid",
            label="{:s}".format(obs["pol_names"][pol_i]),
        )[0]
        for pol_i in range(cal["n_pol"])
    ]

    # Arrange calibration subplots in rows, cols with a maximum per page.
    # Multiple pages will be required beyond 128 stations
    n_rows = 8
    n_cols = 16
    n_pages = int(np.ceil(obs["n_tile"] / (n_cols * n_rows)))

    # Create dictionary for the subplot grid to create reliable, static plots
    gridspec_kw = {
        "left": 0.05,
        "right": 0.98,
        "top": 0.92,
        "bottom": 0.05,
        "hspace": 0.35,
    }

    # Create calibration solution plots for each page of 128 maximum stations for the specified types
    for type_i in range(n_types):
        for page_i in range(n_pages):
            # Calculate number of stations for this page
            if obs["n_tile"] // (n_cols * n_rows * (page_i + 1)) == 0:
                page_tiles = obs["n_tile"] - (n_cols * n_rows * (page_i + 1))
            else:
                page_tiles = n_cols * n_rows

            # Define the subplots, add extra rows on the figsize to account for labels
            fig_amp, ax_amp = plt.subplots(
                n_rows, n_cols, figsize=(n_cols, n_rows + 2), gridspec_kw=gridspec_kw
            )
            fig_phase, ax_phase = plt.subplots(
                n_rows, n_cols, figsize=(n_cols, n_rows + 2), gridspec_kw=gridspec_kw
            )

            # Cycle through each row and col on the page and set tick and labels
            for row_i in range(n_rows):
                for col_i in range(n_cols):
                    # Set fontsize maximum number of ticks on xaxis to 2
                    ax_amp[row_i, col_i].xaxis.set_major_locator(ticker.MaxNLocator(2))
                    ax_phase[row_i, col_i].xaxis.set_major_locator(
                        ticker.MaxNLocator(2)
                    )
                    ax_amp[row_i, col_i].tick_params(labelsize=8, direction="in")
                    ax_phase[row_i, col_i].tick_params(labelsize=8, direction="in")

                    # Only set tick labels on the leftmost column and the bottommost row
                    if col_i > 0:
                        ax_amp[row_i, col_i].set_yticks([])
                        ax_phase[row_i, col_i].set_yticks([])
                    if row_i < 7:
                        ax_amp[row_i, col_i].set_xticks([])
                        ax_phase[row_i, col_i].set_xticks([])

                    # Set the y-axis min and max given the plot type
                    ax_amp[row_i, col_i].set_ylim(
                        amp_minmax[2 * type_i], amp_minmax[2 * type_i + 1]
                    )
                    ax_phase[row_i, col_i].set_ylim(
                        phase_minmax[2 * type_i], phase_minmax[2 * type_i + 1]
                    )

            # Plot each tile on the specified subplot
            tile_start = page_i * (n_cols * n_rows)
            for tile_i in range(tile_start, page_tiles + tile_start):
                col = (tile_i - page_i * (n_cols * n_rows)) % n_cols
                row = (tile_i - page_i * (n_cols * n_rows)) // n_cols

                ax_amp[row, col].set_title(
                    str(obs["baseline_info"]["tile_names"][tile_i]), fontsize=9
                )
                ax_phase[row, col].set_title(
                    str(obs["baseline_info"]["tile_names"][tile_i]), fontsize=9
                )

                for pol_i in range(cal["n_pol"]):
                    if pol_i == 0:
                        colour = "b-"
                    elif pol_i == 1:
                        colour = "r-"

                    # Select current plotting type
                    if type_i == 0:
                        # Gain and phase fit solutions
                        amp = cal_sol_amp[pol_i, :, tile_i].squeeze()
                        phase = cal_sol_phase[pol_i, :, tile_i].squeeze()
                    if type_i == 1:
                        # Gain and phase residuals (per-frequency solutions minus fit solutions)
                        amp = cal_res_amp[pol_i, :, tile_i].squeeze()
                        phase = cal_res_phase[pol_i, :, tile_i].squeeze()
                    if type_i == 2:
                        # Gain and phase per-frequency solutions
                        amp = cal_raw_amp[pol_i, :, tile_i].squeeze()
                        phase = cal_raw_phase[pol_i, :, tile_i].squeeze()

                    # Leave blanks for flagged or undefined tiles
                    if np.isnan([amp, phase]).all() or tile_i not in tile_i_use:
                        ax_amp[row, col].set_axis_off()
                        ax_phase[row, col].set_axis_off()
                    else:
                        ax_amp[row, col].plot(freq_arr_use, amp, colour, lw=1)
                        ax_phase[row, col].plot(freq_arr_use, phase, colour, lw=1)

            fig_amp.suptitle(pyfhd_config["obs_id"], fontsize=14)
            fig_phase.suptitle(pyfhd_config["obs_id"], fontsize=14)

            fig_amp.supylabel("Amplitude")
            fig_amp.supxlabel("Frequency (MHz)")

            fig_phase.supylabel("Phase (radians)")
            fig_phase.supxlabel("Frequency (MHz)")

            fig_amp.legend(
                handles=patches,
                loc="outside upper right",
                ncol=2,
                frameon=False,
                fontsize=12,
                handlelength=1,
            )
            fig_phase.legend(
                handles=patches,
                loc="outside upper right",
                ncol=2,
                frameon=False,
                fontsize=12,
                handlelength=1,
            )

            # Page naming conventions
            if n_pages > 1:
                page_name = "_page" + str(page_i)
            else:
                page_name = ""

            # Save amplitude and phase plot given the type
            fig_amp.savefig(
                f"{save_path_roots[2 * type_i]}{page_name}.png",
                bbox_inches="tight",
                dpi=200,
            )
            fig_phase.savefig(
                f"{save_path_roots[2 * type_i + 1]}{page_name}.png",
                bbox_inches="tight",
                dpi=200,
            )

            plt.close(fig_amp)
            plt.close(fig_phase)

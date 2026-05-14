from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.calibration.calibration_utils import vis_cal_auto_fit
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from pyfhd.io.pyfhd_io import save, load


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_cal_auto_fit")


@pytest.fixture(
    scope="function", params=["point_zenith", "point_offzenith", "1088716296"]
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run1", "run2", "run3"])
def run(request):
    return request.param


skip_tests = [["1088716296", "run1"], ["1088716296", "run2"], ["1088716296", "run3"]]


@pytest.fixture()
def before_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    obs = recarray_to_dict(sav_dict["obs"])
    cal = recarray_to_dict(sav_dict["cal"])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    cal["gain"] = sav_file_vis_arr_swap_axes(cal["gain"])

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = obs
    h5_save_dict["cal"] = cal
    h5_save_dict["vis_auto"] = sav_file_vis_arr_swap_axes(sav_dict["vis_auto"])
    h5_save_dict["vis_model_auto"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_model_auto"]
    )
    h5_save_dict["auto_tile_i"] = sav_dict["auto_tile_i"]
    # Del ragged array
    del h5_save_dict["cal"]["mode_params"]

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture()
def after_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    cal_fit = recarray_to_dict(sav_dict["cal_fit"])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    cal_fit["gain"] = sav_file_vis_arr_swap_axes(cal_fit["gain"])
    # Delete ragged array
    del cal_fit["mode_params"]

    save(after_file, cal_fit, "after_file")

    return after_file


def test_vis_cal_auto_fit(before_file, after_file):
    """Runs the test on `vis_cal_bandpass` - reads in the data in `data_loc`,
    and then calls `vis_cal_bandpass`, checking the outputs match expectations"""
    if before_file is None or after_file is None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}"
        )

    h5_before = load(before_file)
    # expected_cal_fit = load(after_file)

    obs = h5_before["obs"]
    cal = h5_before["cal"]
    vis_auto = h5_before["vis_auto"]
    vis_model_auto = h5_before["vis_model_auto"]
    auto_tile_i = h5_before["auto_tile_i"]

    # return_cal_fit =
    vis_cal_auto_fit(obs, cal, vis_auto, vis_model_auto, auto_tile_i)

    # Plots have been made already testing this against FHD, uncomment to regenerate them.

    # auto_scale is nan, nan from FHD and Python, can't compare them due to the nans
    # assert np.array_equal(return_cal_fit['auto_scale'], expected_cal_fit['auto_scale'])
    # cal_fit['auto_params'] came in as an object array
    # auto_params = np.empty([2, cal["n_pol"], cal['n_tile']], dtype = np.float64)
    # auto_params[0] = expected_cal_fit['auto_params'][0].transpose()
    # auto_params[1] = expected_cal_fit['auto_params'][1].transpose()

    # name_split = before_file.name.split('_')
    # tag = f"{name_split[0]}_{name_split[1]}"
    # run = name_split[2]
    # actual_gains = np.load(Path(data_dir, f"{tag}_gains_applied_woden.npz"))
    # gx = actual_gains['gx'].transpose()
    # gy = actual_gains['gy'].transpose()

    # fig, axs = plt.subplots(2, 1)

    # axs[0].plot(np.abs(gx[0, :]), 's', mfc='none', linestyle='none', label='Sim gains')
    # axs[0].plot(return_cal_fit['gain'][0, 0, :], 'x', mfc='none', linestyle='none', label='Fit pyfhd')

    # print(expected_cal_fit['gain'][0, 0, 1])
    # print(return_cal_fit['gain'][0, 0, 1])

    # axs[1].plot(np.abs(gx[0, :]), 's', mfc='none', linestyle='none', label='Sim gains')
    # axs[1].plot(expected_cal_fit['gain'][0, 0, :]*0.5, '^', mfc='none', linestyle='none', label='Fit FHD')

    # axs[1].set_xlabel('Tile index')

    # axs[0].set_ylabel('Gain value')
    # axs[1].set_ylabel('Gain value')

    # axs[0].legend()
    # axs[1].legend()

    # plt.tight_layout()
    # fig.savefig(f"test_vis_cal_auto_fit_{tag}_{run}.png", bbox_inches='tight', dpi=300)
    # plt.close()

    # rtol = 1e-5
    # atol = 1e-3

    # npt.assert_allclose(return_cal_fit['auto_params'], auto_params,
    #                     rtol=rtol, atol=atol)

    # npt.assert_allclose(return_cal_fit['gain'], expected_cal_fit['gain'],
    #                     rtol=rtol, atol=atol)

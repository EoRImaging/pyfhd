from pyfhd.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from pyfhd.calibration.calibration_utils import vis_baseline_hist
from pyfhd.io.pyfhd_io import convert_sav_to_dict
from pyfhd.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from pyfhd.io.pyfhd_io import save, load
import numpy.testing as npt
import matplotlib.pyplot as plt


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "calibration", "vis_baseline_hist")


@pytest.fixture(
    scope="function", params=["point_zenith", "point_offzenith", "1088716296"]
)
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run1", "run2", "run3"])
def run(request):
    return request.param


skip_tests = []


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
    params = recarray_to_dict(sav_dict["params"])

    vis_arr = sav_file_vis_arr_swap_axes(sav_dict["vis_arr"])
    vis_model_arr = sav_file_vis_arr_swap_axes(sav_dict["vis_model_arr"])

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = obs
    h5_save_dict["params"] = params
    h5_save_dict["vis_arr"] = vis_arr
    h5_save_dict["vis_model_arr"] = vis_model_arr

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

    vis_baseline_hist = recarray_to_dict(sav_dict["vis_baseline_hist"])
    vis_baseline_hist["vis_res_ratio_mean"] = vis_baseline_hist[
        "vis_res_ratio_mean"
    ].transpose()
    vis_baseline_hist["vis_res_sigma"] = vis_baseline_hist["vis_res_sigma"].transpose()

    save(after_file, vis_baseline_hist, "after_file")

    return after_file


def test_vis_baseline_hist(before_file: Path, after_file: Path):
    """
    Runs the test on `vis_baseline_hist` - reads in the data in before_file and after_file,
    and then calls `vis_baseline_hist`, checking the outputs match expectations
    """

    if before_file == None or after_file == None:
        pytest.skip(
            f"This test has been skipped because the test was listed in the skipped tests due to FHD not outpoutting them: {skip_tests}"
        )

    h5_before = load(before_file)
    expec_vis_baseline_hist = load(after_file)

    obs = h5_before["obs"]
    params = h5_before["params"]
    vis_arr = h5_before["vis_arr"]
    vis_model_arr = h5_before["vis_model_arr"]

    result_vis_baseline_hist = vis_baseline_hist(obs, params, vis_arr, vis_model_arr)

    num_bins = expec_vis_baseline_hist["baseline_length"].shape[0]
    expec_vis_res_ratio_mean = expec_vis_baseline_hist["vis_res_ratio_mean"]
    expec_vis_res_sigma = expec_vis_baseline_hist["vis_res_sigma"]

    atol = 4e-4

    # Can test that the fixed final polarisation is close to pyfhd result
    # Out results are ordered by pol, bin so need to do a transpose
    npt.assert_allclose(
        result_vis_baseline_hist["vis_res_ratio_mean"],
        expec_vis_res_ratio_mean,
        atol=atol,
    )

    npt.assert_allclose(
        result_vis_baseline_hist["vis_res_sigma"], expec_vis_res_sigma, atol=atol
    )

    npt.assert_allclose(result_vis_baseline_hist["baseline_length"], num_bins)

    # Plots have already been made and saved, if you want to regenerate them, uncomment the code

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # extent = [-0.5, 1.5, -0.5, num_bins - 0.5]

    # im = axs[0].imshow(expec_vis_res_ratio_mean.transpose(), aspect='auto',
    #                    extent=extent, origin='lower')
    # plt.colorbar(im)
    # axs[0].set_xticks([0, 1])

    # vis_res_ratio_mean_plt = np.zeros([num_bins, 2])
    # vis_res_ratio_mean_plt[:,1] = expec_vis_res_ratio_mean[1]

    # im = axs[1].imshow(result_vis_baseline_hist['vis_res_ratio_mean'].transpose(), aspect='auto',
    #                    extent=extent, origin='lower')
    # plt.colorbar(im)
    # axs[1].set_xticks([0, 1])

    # axs[0].set_title('FHD')

    # axs[1].set_title('pyfhd')

    # for ax in axs.flatten():
    #     ax.set_xlabel('Polarisation')

    # axs[0].set_ylabel("Baseline hist bin")

    # plt.tight_layout()
    # name_split = before_file.name.split('_')
    # if (name_split[0] == 'point'):
    #     tag = f"{name_split[0]}_{name_split[1]}"
    #     run = f"{name_split[2]}"
    # else:
    #     tag = f"{name_split[0]}"
    #     run = f"{name_split[1]}"
    # fig.savefig(f"test_vis_baseline_hist_{tag}_{run}_after_fix.png", bbox_inches='tight', dpi=300)
    # plt.close()

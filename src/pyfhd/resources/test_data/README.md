# Running pyfhd Small Data Tests

In order to run the tests that will have the test data storeed in this directory you need to do the following:

1. Download the data, you can do so here: [![Static Badge](https://img.shields.io/badge/Test%20Data%20DOI-10.5281%2Fzenodo.15687722-grey?labelColor=blue)](https://doi.org/10.5281/zenodo.15687722). Put the zip file
in the following directory inside the repository (the directory given is relative to the root of the repository): `src/pyfhd/resources/test_data`

2. Unzip the zip file in the previously mentioned `src/pyfhd/resources/test_data` directory.

3. In a terminal, at the root of the repository, run the following command:

```bash
pytest -m github_actions
```

You should expect output that looks like this:

```bash
=========================================== test session starts ===========================================
platform linux -- Python 3.12.7, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/skywatcher/projects/pyfhd
configfile: pyproject.toml
plugins: metadata-3.1.1, cov-6.1.1, html-4.1.1
collected 436 items / 315 deselected / 121 selected

tests/test_calibration/test_cal_auto_ratio_divide.py .....s                                         [  4%]
tests/test_calibration/test_cal_auto_ratio_remultiply.py .....s                                     [  9%]
tests/test_calibration/test_calculate_adaptive_gain.py ....                                         [ 13%]
tests/test_calibration/test_vis_cal_bandpass.py ........s                                           [ 20%]
tests/test_calibration/test_vis_cal_polyfit.py ........s                                            [ 28%]
tests/test_calibration/test_vis_calibration_flag.py .....s..s                                       [ 35%]
tests/test_data_setup/test_sample_data_extraction.py .                                              [ 36%]
tests/test_gridding/test_gridding_utils/test_interpolate_kernel.py ...                              [ 38%]
tests/test_gridding/test_visibility_grid.py ......                                                  [ 43%]
tests/test_healpix/test_healpix_cnv_apply.py ssssss......                                           [ 53%]
tests/test_healpix/test_healpix_cnv_generate.py s.                                                  [ 55%]
tests/test_io/test_configuration.py .                                                               [ 56%]
tests/test_io/test_save_and_load.py ...                                                             [ 58%]
tests/test_pyfhd_tools/test_array_match.py ...                                                      [ 61%]
tests/test_pyfhd_tools/test_deriv_coefficients.py ...                                               [ 63%]
tests/test_pyfhd_tools/test_histogram.py .............                                              [ 74%]
tests/test_pyfhd_tools/test_meshgrid.py ...                                                         [ 76%]
tests/test_pyfhd_tools/test_rebin.py ...................                                            [ 92%]
tests/test_pyfhd_tools/test_region_grow.py ..                                                       [ 94%]
tests/test_pyfhd_tools/test_resistant_mean.py ......                                                [ 99%]
tests/test_pyfhd_tools/test_weight_invert.py .                                                      [100%]

============================================ warnings summary =============================================
tests/test_calibration/test_cal_auto_ratio_divide.py::test_cal_auto_ratio_divide[1088716296-run1]
  /home/skywatcher/projects/pyfhd/src/pyfhd/pyfhd_tools/pyfhd_utils.py:715: RuntimeWarning: overflow encountered in divide
    result[i_use] = 1 / weights[i_use]

tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[point_zenith-run1]
tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[point_zenith-run3]
tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[point_offzenith-run1]
tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[point_offzenith-run3]
tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[1088716296-run1]
  /home/skywatcher/projects/pyfhd/src/pyfhd/calibration/calibration_utils.py:816: RuntimeWarning: divide by zero encountered in divide
    gain3[freq_use, :] /= gain2_input[freq_use, :]

tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[point_zenith-run1]
tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[point_offzenith-run1]
tests/test_calibration/test_vis_cal_bandpass.py::test_vis_cal_bandpass[1088716296-run1]
  /home/skywatcher/projects/pyfhd/src/pyfhd/calibration/calibration_utils.py:816: RuntimeWarning: invalid value encountered in divide
    gain3[freq_use, :] /= gain2_input[freq_use, :]

tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_zenith-run1]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_zenith-run2]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_zenith-run3]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_offzenith-run1]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_offzenith-run2]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_offzenith-run3]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[1088716296-run1]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[1088716296-run2]
  /home/skywatcher/projects/pyfhd/src/pyfhd/calibration/calibration_utils.py:1164: RuntimeWarning: divide by zero encountered in divide
    gain_arr = og_gain_arr[pol_i] / cal["gain"][pol_i]

tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_zenith-run1]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_zenith-run2]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_zenith-run3]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_offzenith-run1]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_offzenith-run2]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[point_offzenith-run3]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[1088716296-run1]
tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[1088716296-run2]
  /home/skywatcher/projects/pyfhd/src/pyfhd/calibration/calibration_utils.py:1164: RuntimeWarning: invalid value encountered in divide
    gain_arr = og_gain_arr[pol_i] / cal["gain"][pol_i]

tests/test_calibration/test_vis_cal_polyfit.py::test_vis_cal_polyfit[1088716296-run1]
  /home/skywatcher/projects/pyfhd/src/pyfhd/calibration/calibration_utils.py:1193: RuntimeWarning: invalid value encountered in divide
    norm_autos = auto_ratio[pol_i] / rebin(

tests/test_gridding/test_visibility_grid.py::test_visibility_grid[4]
  /home/skywatcher/projects/pyfhd/tests/test_gridding/test_visibility_grid.py:158: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
    new_arr[0] = h5_before["fi_use"]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 108 passed, 13 skipped, 315 deselected, 27 warnings in 36.53s ======================
```

These are the same tests that the actions run on every pull request and push to main.

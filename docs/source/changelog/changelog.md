# Change Log

## Unreleased

### Breaking Changes!
* UVBeam files must now be passed using the `uvbeam-file-path` option, which
can take any path on the computer, rather than placing them in a folder in the
repo and passing None to `beam_file-path`.
* The `beam-file-path` configuration option has been renamed to `saved-beam-file-path`
to avoid confusion with the new `uvbeam-file-path` option.

### New Features
* UVBeams are fully and properly integrated. Beam decomposition is implemented
in pyuvdata.UVBeam and just called directly.
* New `uvbeam-file-path` and `uvbeam-freq-buffer` options in configuration.
* Analytic beams (subclassed from pyuvdata.AnalyticBeam) are now supported. Use
the new `analytic-beam-yaml` option to configure them.
* Added handling for `~` in paths in config yamls.

### Bug Fixes
* The telescope location is now extracted from the uvfits antenna table, rather
than obtained from astropy's site list, preventing errors when the telescope
is not listed in astropy's site list.
* Fixed a bug in `beam_utils.beam_power` where the beam phase was calculated
improperly, resulting in NaNs.
* Fixed a bug in setting up the gridding kernel that resulted in the kernel not
being centered on the uv space beam.
* Fixed a bug where some pixels slightly below the horizon could be kept,
resulting in beam interpolation errors.
* Fixed a bug in beam setup where the FFT direction was wrong, resulting in
incorrect beam normalization.
* Fixed errors in the calculation of the beam squared area.
* Fixed errors in converting antenna numbers to indices in `obs.create_obs`.
* Fixed errors in calibration plotting with more than 128 antennas.
* Fixed a bug when `split-ps-export` is set to False that resulted in gridding no data.
* Fixed some bugs when `restrict-healpix-inds` is False that caused several errors.
* Removed unimplemented image weighting options in `pyfhd_setup` that resulted in
files claiming to have different image weighting than they actually had.
* Fixed tile height calculation in the case where there is not a metafits file,
e.g. for telescopes other than the MWA.
* Fixed some indexing errors with newer versions of numpy in `vis_calibrate_subroutine`
and `healpix_utils.healpix_cnv_generate`.
* Fixed a problem in `pyfhd_io.save_dataset` where short antenna names caused
h5py errors.
* Fixed a bug in FITS file writing that caused deprecation warnings from astropy.
* Fixed checkpointing to actually work.
* Fixed a bug in the uvfits reader where it assumed the presence of "ra" and 
"dec" header items which often present in MWA uvfits files but are non-standard.
* Fixed a bug with newer versions of numpy.
* Fixed `vis_model_transfer` to handle standard IDL FHD folder structure.
* Fixed a bug in `calibration_utils.vis_baseline_hist` where flagging wasn't
fully propagated, causing shape errors.
* Fixed a bug that could cause a KeyError in `pyfhd.main`.
* Fixed a bug that could cause a KeyError in `beam_utils.beam_image`.
* Fixed a bug that caused a numpy indexing error with newer versions of numpy in  
`beam_utils.beam_image`.
* Fixed the a bug caused by a bad default value of `baseline_threshold` in 
`gridding_utils.dirty_image_generate`.
* Fixed spelling errors in example config yamls.
* Fixed a bug that could cause an undefined variable error in image plotting.

### Test Changes
* Fixed two tests in `test_weight_invert` to handle floating point errors.
* Fixed fixtures in `test_vis_model_transfer` and `test_quickview` to handle
computer-specific paths in test data.


### Dependency Changes

### Version Changes

### Translation Changes

## 1.0.2

### New Features
* Docs point to github pages for the github action results

### Bug Fixes
* vis_baseline_hist option actually used

### Test Changes
* Quickview tests have issues not previously present with HEALPIX fits files

### Version Changes
* Bump to 1.0.2


## Version 1.0

PyFHD 1.0.1 🎉

The translation from FHD (IDL) to PyFHD (Python) is now mostly complete and runs without needing IDL.

In terms of the FHD pipeline that has been translated we'll go through it bit by bit:

1. PyFHD uses `configargparse` and `yaml` to set all the options for a run, all the options have help text associated with them, coming from the FHD dictionary or comments in the code. The initial setup also setups a logging system using Python's inbuilt logging system giving you control on whether you see the log in the terminal, in a file, or both or neither. 
2. Extracting visibilities data, weights, and parameters,has been fully translated with the exception that the shape of visiblities in general is `[number of polarizations, number of frequencies, number of baselines]` to accomodate the fact that Python is row based vs column based in IDL. The shape change made it easier to translate the indexing across to PyFHD, as the indexing could be translated directly, if you wish to import PyFHD visibilities into FHD at some point, you will need to transpose the baselines and frequencies. 
3. The observation metadata structure/dictionary and the antenna layout structure/dictionary has been recreated as it was in FHD with small changes in names where it made sense and/or values due to the use of libraries like `astropy`.
4. Much of the `beam_setup` has been translated using a combination of `pyuvdata` and translation of FHD in an effort to not have it be too MWA specific, feel free to experiment with it as it has not been tested and will need additional work (and potential research to best replace some functions usd in FHD). The ability to import a beam from a `sav` file or `HDF5` file has been done, however it is going to expect the file to have the same structure as the `psf` structure in `FHD`. Furthermore, the beam HDF5 can be lazy loaded to reduce memory use at the expense of some performance in gridding (takes twice as long in tests). 
5. Basic Flagging has been translated directly into PyFHD, and `vis_source_model` hasn't been translated as other libraries like WODEN are good for creating visibility models. PyFHD only has the ability to import a visibility model and then flag that visibility model. The Galaxy models structures aren't in PyFHD either, if you want them, you will need to translate them or add them in.
6. The entirety of calibration from `FHD` has been translated into `PyFHD`, I don't think a single thing hasn't been translated unless it was completely undocumented with no explanations of its existence or evidence of its use as was the case with some options.
7. Flagging, noise calculation and the updating of the visibility weights post-calibration and pre-gridding has been translated as well
8. Gridding has been translated fully with exception to the mapping function as it was unclear at the time what the best solution was for the sparse matrices that were required. It is clear now the mapping function could be done with a HDF5 file and making sure it can be lazy loaded with chunking with `h5py`, if you wish to have the mapping function in an effort to translate the deconvolution part of the `FHD` pipeline, hope this helps.
9. The `fhd_quickview` has many of the pieces left out given you could have the capability to practically skip the whole pipeline and almost re-run the whole thing in just `fhd_quickview`. As such `pyfhd_quickview` focuses more on the saving of the final visibilities, results of the whole run including gridding and calibration. `pyfhd_quickview` also creates the dirty fits files but not including the stokes due to time constraints. `pyfhd_quickview` also doesn't create much of the plots that are in `fhd_quickview` as many are for diagnostics and may no longer be useful, if you wish to make them, create an option in the configuration and make the plot with explicit reasons for it to exist.
10. `healpix_snapshot_cube_generate` has been translated fully with the ability to make HDF5 Healpix files which should be compatible with any relevant FHD and IDL tools if need be given that IDL has capabilities to read in HDF5 files. With that said, the `vis_model_freq_split` function does not pass its tests, which is the heart of the HEALPIX generation, so if you wish to fix it, please go ahead.

The PyFHD pipeline also has a checkpointing system so you can save checkpoints after creating the `obs` dictionary, calibration and gridding so you can load up previous points and run again if you get a failure after a major processing step. A great example would be if you used a wrong option for gridding or got a failure but the calibration ran fine, you could load the calibration checkpoint and start gridding again.

PyFHD will output everything into a single directory from one run, containing a directory structure itself, please refer to the docs for this directory structure. The docs will also detail the required inputs. Find the docs [here](https://pyfhd.readthedocs.io/en/latest/).

That should cover the major pieces of the FHD pipeline, other notable pieces of work in `PyFHD` is the replication of IDL's:
* `HISTOGRAM` function with the making of `REVERSE_INDICES` using Numba it's able to produce a histogram and reverse indices array for 1 billion integers in 7 seconds.
* `REBIN` function - Done completely with `NumPy`
* `REGIN_GROW` function - Done with a combination of `NumPy` and `SciPy`
* `RESISTANT_MEAN` function - Done with `NumPy`

Almost every function that has been translated from FHD or IDL has many tests alongside of it, giving PyFHD a total of 431 tests to give you some confidence that PyFHD does actually match what FHD does down to single precision.

Some bugs that did exist in FHD have been fixed during the translation to PyFHD, these has mostly been passed along and been fixed in `FHD`, there are some mysteries like that of `vis_cal_auto_fit` which seem to work better in PyFHD during tests but we're not sure why.

PyFHD has been tested to run with Python `3.10+` with the following packages as dependencies:

```python
"astropy>=6.1.7",
"colorama>=0.4.6",
"configargparse>=1.7",
"h5py>=3.13.0",
"healpy>=1.18.1",
"importlib-resources>=6.5.2",
"matplotlib>=3.10.3",
"numba>=0.61.2",
"numpy>=2.2.5",
"pyuvdata>=3.2.1",
"scipy>=1.15.3"
```

The following packages are development dependencies (if you're developing features on PyFHD):

```python
"black>=25.1.0",
"ipykernel>=6.29.5",
"myst-parser>=4.0.1",
"pip>=25.1.1",
"pre-commit>=4.2.0",
"pytest>=8.3.5",
"pytest-cov>=6.1.1",
"sphinx>=8.1.3",
"sphinx-argparse>=0.5.2",
"sphinx-rtd-theme>=3.0.2",
```

PyFHD has been built using `uv` utilising `pyproject.toml` and has been published to PyPi can be installed via `pip`.

```bash
pip install pyfhd
```

As the developer (Joel Dunstan, [SkyWa7ch3r](https://github.com/SkyWa7ch3r)), I'd like to thank Nichole Barry for giving me the chance to work on this translation for many years, it has been rewarding (though sometimes frustrating and challenging!) work. I'd also like to thank Jack Line who has been crucial at many times working on PyFHD directly with regards to testing and the translation of the source modelling. I'd also like to thank Bryna Hazelton for her advice during the translation effort and providing examples of the use of `pyuvdata` for the `beam_setup`.

If you need to reach me (Joel), I'm on the EoR Analysis slack channel, see you around.
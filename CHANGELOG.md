# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.4.7]
### Fixed
* [617](https://github.com/dbekaert/RAiDER/issues/617) - RAiDER created `.netrc` is too permissive

## [0.4.6]

### Added
* Adds an `s1_orbits.py` module which includes:
  * `get_orbits_from_slc_ids` to download the associated orbit files for a list of Sentinel-1 SLC IDs
  * `ensure_orbit_credentials` to ensure ESA CSDE credentials have been provides to download orbit files. This should be called before `sentineleof` is used to download orbits.
* Adds a `setup_from_env` function to `models/credentials.py` which will pull *all* credentials needed for acquiring weather model data from environment variables and ensure the correct config file is written. This makes setting up credentials in CI pipelines significantly easier

### Changed
* `sentineleof` upgraded to version 0.9.5 or later to (a) fetch orbits from ESA CDSE and (b) ensure that if CDSE fetch fails, code resorts to ASF orbit repository

### Fixes
* RAiDER is now tested on Python version 3.9-3.12
* All typehints are now Python 3.9 compatible
* [607](https://github.com/dbekaert/RAiDER/issues/607): Python entrypoint loading is now compatible with Python 3.12
* [610](https://github.com/dbekaert/RAiDER/issues/610): Sentinel-1 orbit availability due to ESA migrating Sentinel-1 orbit files from Copernicus Open Access Hub (Scihub) to the new  Copernicus Data Space Ecosystem (CDSE)
* make weather file directory when it doesn't exist
* Ensures the `models/data/alaska.geojson.zip` file is packaged when building from the source tarball
* Make ISCE3 an optional dependency in `s1_azimuth_timing.py`
+ Added unit tests and removed unused and depracated functions

### Removed
* `hyp3lib`, which was only used for downloading orbit fies, has been removed in favor of `sentineleof`

## [0.4.5]

### Fixes
* [#583](https://github.com/dbekaert/RAiDER/issues/583): it appears that since the issues with geo2rdr cropped up during our processing campaign, there has been a new release of ISCE3 that resolves these failures with `geo2rdr` and the time interpolation that uses this ISCE3 routine.
* [#584](https://github.com/dbekaert/RAiDER/issues/584): failed Raider step function in hyp3 job submission when HRRR model times are not available (even within the valid model range) - to resolve, we check availability of files when delay workflow called with a) azimuth_grid_interpolation and b) input to workflow is GUNW. If weather model files are unavailable and the GUNW is on s3, do nothing to GUNW (i.e. do not add tropo delay) and exit successfully. If weather model files are unavailable and the GUNW is on local disk, raise `ValueError`
* [#587](https://github.com/dbekaert/RAiDER/issues/587): similar to 584 except added here to the mix is control flow in RAiDER.py passes over numerous exceptions in workflow. This is fixed identically as above.
* [#596](https://github.com/dbekaert/RAiDER/issues/596): the "prefix" for aws does not include the final netcdf file name, just the sub-directories in the bucket and therefore extra logic must be added to determine the GUNW netcdf file name (and the assocaited reference/secondary dates). We proceed by downloading the data which is needed regardless. Tests are updated.

### Removed
* Removes `update` option (either `True` or `False`) from calcGUNW workflow which asks whether the GUNW should be updated or not. In existing code, it was not being used/applied, i.e. previous workflow always updated GUNW. Removed input arguments related from respective functions so that it can be updated later.

### Added
* Allow for Hyp3 GUNW workflow with HRRR (i.e. specifying a gunw path in s3) to successfully exit if any of the HRRR model times required for `azimuth-time-grid` interpolation (which is default interpolatin method) are not available when using bucket inputs (i.e. only on the cloud)
* For GUNW workflow, when model is HRRR, azimuth_time_grid interpolation used, and using a local GUNW, if requisite weather model files are not available for  raise a ValueError (before processing)
* Raise a value error if non-unique dates are given in the function `get_inverse_weights_for_dates` in `s1_azimuth_timing.py` 
* Added metadata provenance for each delay layer that is included in GUNW and the cube workflow generally in `calcDelays` including:
   * `model_times_used` - the weather models used and interpolated for the delay calculation
   * `interpolation_method` - whether `none`, `center_time`, or `azimuth_time_grid` methods were used - see description in `calcDelayGUNW`
   * `scene_center_time` - the center time in which the associated SAR image was acquired
* Stages GMAO data for GUNW testing of correct dataset update i.e. in the test `test_GUNW_dataset_update`.
* Stages HRRR data for `test_HRRR_ztd` test.
* Ensures ISCE3 is `>=0.15.0`
* Uses correct HyP3 S3 prefix conventions and filename suffix within test patches to improve readability of what tests are mocking (see comments in #597).

### Changed
* Get only 2 or 3 model times required for azimuth-time-interpolation (previously obtained all 3 as it was easier to implement) - this ensures slightly less failures associated with HRRR availability. Importantly, if a acquisition time occurs during a model time, then we order by distance to the reference time and how early it occurs (so earlier times come first if two times are equidistant to the aquisition time).
* Made test names in `test_GUNW.py` more descriptive
* Numpy docstrings and general linting to modified function including removing variables that were not being accessed
* hrrr_download to ensure that `hybrid` coordinate is obtained regardless how herbie returns datacubes and ensures test_HRRR_ztd passes consistently
   * Remove the command line call in `test_HRRR_ztd.py` and call using the python mock up of CLI for better error handling and data mocking.
* Return xarray.Dataset types for RAiDER.calcGUNW.tropo_gunw_slc and RAiDER.raider.calcDelayGUNW for easier inspection and testing
* Fixes tests for checking availability of HRRR due Issue #596 (above).

## [0.4.4]
* For s1-azimuth-time interpolation, overlapping orbits when one orbit does not cover entire GUNW product errors out. We now ensure state-vectors are both unique and in order before creating a orbit object in ISCE3.

## [0.4.3]
+ Bug fixes, unit tests, docstrings
+ Prevent ray tracing integration from occuring at exactly top of weather model
+ Properly expose z_ref (max integration height) parameter, and dont allow higher than weather model
+ Min version for sentineleof for obtaining restituted orbits.
+ Rename datetime columns and convert from strings for GNSS workflow
+ Use native model levels in HRRR which extend up to 2 hPa as opposed to 50 hPa in pressure levels
+ Update tests to account for different interpolation scheme
+ Dont error out when the weather model contains nan values (HRRR)
+ Fix bug in fillna3D for NaNs at elevations higher than present in the weather model
+ write delays even if they contain nans
+ check that the aoi is contained within HRRR extent
+ streamline some unit tests to remove downloading
+ move the ray building out of the _build_cube_ray and into its own function for cleaner testing
+ update the tests to use the new build_ray function
+ If the processed weather file exists use it; otherwise check if raw exists and covers study area; otherwise download new
+ Update the integration height for raytracing from 50 km to 80 km
+ Reinstate test 3 (slant proj and ray trace), remove unused calls with ZREF
+ Add buffer to W/E for ERA5
+ refactor imports to allow for a cleaner raider-base
+ Add buffer to HRES when downloading as with the other models
+ Refactor to pass a weather file directly to fetch
+ Update staged weather models to reflect update to aligned grid
+ Correctly pass buffered bounds when aligning grid
+ Check the valid bounds prior to starting and use HRRR-AK if its correct so that rounding times to obtain data at are correctly fed to Herbie
+ Update test_intersect to already existing weather model files 
+ Replace the real weather model files used for the synthetic test with the correct ones (bounding box changed slightly)
+ Update test_scenerio_1 to match golden data by selecting a grid by lat/lon rather than indices
+ Adjust the buffering to account for grid spacing
+ Update ERA5 model coordinates to reflect changes in support of HRRR
+ Re-work the HRRR weather model to use herbie (https://github.com/blaylockbk/Herbie) for weather model access. HRRR conus and Alaska validation periods are respectively 2016-7-15 and 2018-7-13 onwards.
+ minor bug fixes and unit test updates
+ add log file write location as a top-level command-line option and within Python as a user-specified option
+ account for grid spacing impact on bounding box before downloading weather model
+ update the GUNW test to account for change in grid spacing on affine transform
+ add CLI for the old processDelayFiles script and rename to raiderCombine
+ Fix gridding bug in accessing HRRR-AK 
+ misc clean-up
+ Specify unbuffered python output in the docker entrypoint script using `python -um RAiDER.cli ...` whose `__main__.py` is the desired entrypoint.
+ For the GUNW workflow uses azimuth time interpolation using ISCE3 geo2rdr (see [here](https://github.com/ACCESS-Cloud-Based-InSAR/s1_azimuth_time_grid)).
    - Updates `interpolate_time` options to: `'none'` (formerly `False`), `'center_time'` (formerly `True` and `default`), and `azimuth_time_grid` (not implemented previously)
+ Series of bug-fixes/compatibility updates with stats class: 
    + Inconsistent definition of index IDs, which leads to key errors as so when querying the grid space for valid data points
    + Turn off default behavior of plotting minor ticks on colorbars, which translates to unreadable plots especially when scientific notation is involved
    + Assign valid geotrans to output tif files used for replotting/dedup.
    + Properly load existing grids for replotting runs. Before the program crashed as single bands were incorrectly being read as cubes.
    + Update in pandas not backwards compatible with original conditional logic. Specifically, conditions like `(not self.df['Date'].dt.is_leap_year)` replaced with `(self.df['Date'].dt.is_leap_year is False)`
+ add unit tests for the hydro and two pieces of wet equation
+ bump  bottom/top height of user requested levels by ~1mm during ray tracing to ensure interpolation works
+ ensure directories for storage are written
+ fix bug in writing delays for station files
+ Force lat/lon/hgt to float32 so that they line up correctly in stitching
+ Add two stage buffer;
    + first pad user bounding box such that a 3D cube is generated that at min covers user area of interest.
    + then if ray tracing is used, pad the downloaded model in look direction. Assumes look angle is fixed increases with latitude.
+ Update and convert user given AOI to weather model projection (except for HRRR)
+ Clean up error messagse, skip date if temporal interpolation fails
+ Update valid range for ERA5 (current date - 3 months) & ERA5T
+ Temporal interpolation of delays if the requested datetime is more than _THRESHOLD_SECONDS away from the closest weather model available time and `interpolate_time = True` (default behavior)
+ Add assert statement to raise error if the delay cube for each SAR date in a GUNW IFG is not written
+ Verify some constants / equations and remove the comments questioning them
+ Relocate the time resolution of wmodels to one spot
+ Skip test_scenario_3 until a new golden dataset is created
+ Update environment with scipy version minimum and requiring pybind11 (the latter for Apple ARM64 with rosetta2)
+ For GUNW entrypoint and associated workflow, update json metadata when bucket argument is provided to include `weather_model` key with value being a list.
   - A GUNW is downloaded from a bucket prefix with netcdf, json, and png whose name is the <GUNW_ID>. We download the json and update it to be consistent with ASF DAAC schema (link: https://github.com/asfadmin/grfn-ingest/blob/test/verify/src/metadata_schema.json)
+ For the GUNW workflow:
   - Updated GUNW workflow to expose input arguments (usually passed through command line options) within the python function for testing
   - Include integration test of HRRR for GUNW workflow
   - Test the json write (do not test s3 upload/download) in that it conforms to the DAAC ingest schema correctly - we add a weather model field to the metadata in this workflow
   - Removed comments in GUNW test suite that were left during previous development
   - If a bucket is provided and the GUNWs reference or secondary scenes are not in the valid range, we do nothing - this is to ensure that GUNWs can still be delivered to the DAAC without painful operator (i.e. person submitting to the hyp3 API) book-keeping

## [0.4.2]

### New/Updated Features
+ `prepFromGUNW` reads the date/time from the SLCs rather than the GUNW filename
+ `calcDelaysGUNW` allows processing with any supported weather model as listed in [`RAiDER.models.allowed.ALLOWED_MODELS`](https://github.com/dbekaert/RAiDER/blob/dev/tools/RAiDER/models/allowed.py).
+ Removed NCMR removed from supported model list till re-tested
+ `credentials` looks for weather model API credentials RC_file hidden file, and creates it if it does not exists
+ Isolate ISCE3 imports to only those functions that need it.
+ Small bugfixes and updates to docstrings
+ Only orbit file is used (even if multiple specified) to minimize errors and ensure consistency over region
+ GUNW packaging is restructed to store SLC (ref and sec) wet and tropo delays rather than the differential
+ padding made consistent throughout and default arguments reduced (manually update in test_losreader)

## [0.4.1]

### New/Updated Features
+ Reorder target points for intersection
+ Use exact coordinates of DEM to interpolate heights to target lat/lons
+ Support DEM interpolation to station file
+ Implement end-to-end test for intersection of cube with lat/lon files
+ Implement end-to-end test for calculation at stations delay
+ Update AOI to store the output directory so DEM is written to right place
+ `calcDelaysGUNW` will optionally download a GUNW product from AWS S3, process it, and upload a new version of the GUNW product in-place with the tropospheric correction layers added so that RAiDER can be used in ARIA GUNW production via HyP3. **Importantly**, tropospheric correction of GUNW products is still being activitely developed; this workflow, as well as the correction itself, is subject to change.

### Fixed
+ Package data is more explicitly handled so that it is included in the conda-forge build; see [#467](https://github.com/dbekaert/RAiDER/pull/467)

## [0.4.0]

Adding of new GUNW support to RAiDER. This is an interface delivery allowing for subsequent integration into HYP3 (input/output parsing is not expected to change; computed data is not yet verified).

### New/Updated Features
+ Working GUNW entry point in workflow for raider.py
+ Ability to parse a GUNW to workflows from which all required RAiDER information is extracted (e.g. dates, UTC, orbit, bbox, look direction, wavelength) with an option to specify weather model (those already supported by RAiDER) and ability to squeeze in the derived output into the original GUNW product.
+ Delays for GUNW are calculated in RAiDER using the ray-tracing option specifying bbox (GUNW driven), a hardcoded lateral posting (0.05º for HRRR and 0.1º for others),  fixed vertical height levels, using an different orbit file for  secondary and master.
     - The hard-coded heights and posting will be refined per model and to ensure stitching abilities in ARIA-tools.
     - The orbit should be refined to not change between secondary and reference to avoid issues. See https://github.com/dbekaert/RAiDER/discussions/435#discussioncomment-4392665
+ Bug fix for raider.py "date" input argument when multiple dates are requested (i.e. support of requesting two dates or two dates with a sampling).
+ Add unit test for date input argument checking (single day, two dates, two dates with samples)
+ Write the diagnostic weather model files to the 'output_directory' rather than PWD
+ Fix for incorrectly written hard-cored projection embedded in the computed output data
+ Allow for multiple orbits files/dates to be used for slant:projection
+ correctly pass llh to lla_to_ecef function for slant:projection
    ++ verified this doesnt change anything
+ removed deprecated ray projection functionality
+ added 1º buffer for zenith and projected (already done for ray tracing)
+ differential delay is rounded to model-dependent nearest hour
+ version 1c hardcoded into the updated GUNW

### Added dependencies for:
+ sentinelof: used to fetch the orbit for GUNW
+ rioxarray: used for reading rasters with xarray

### Not implemented / supported in this release### Not implemented / supported in this release
+ no temporal interpolation
+ no refined model specific hardcoded spacing and heights
+ no ability for single orbit Interferometric calculation
+ no verification of results

## [0.3.1]
Fixes some missing imports and typing statements

## [0.3.0]
RAiDER package was refactored to expose the main functionality as a Python library, including the `prepareWeatherModel`
and `tropo_delay` functions, as well as anciliarry functions needed for defining AOIs, look vectors, etc.

### New/Updated Features
+ Python library access to main functions for accessing weather model data and calculating delays
+ Slant delay calculation through projection is supported for cubes with orbit files
+ Upgrade dem-stitcher to [`>=2.3.1`](https://github.com/ACCESS-Cloud-Based-InSAR/dem-stitcher/blob/dev/CHANGELOG.md#231) so that the updated urls for the GLO-30 DEM are used.
+ `raider.py ++calcDelaysGUNW GUNWFILE` is enabled as a placeholder only.
+ Upgraded ISCE3 to `>=v0.9.0` to fix a conda build issue as described in [#425](https://github.com/dbekaert/RAiDER/issues/425)
+ Allow user to specify --download_only or download_only=True in the configure file
+ Added documentation for the Python library interface.
+ Added some unit tests.
+ Fixed some bugs and tweaked the CLI.
+ Added unit tests, docstrings, initial API reference
+ __main__ file to allow calls to different functionality. `raider.py ++process downloadGNSS ...` can now perform the functionality of `raiderDownloadGNSS.py ...



## [0.2.0]

RAiDER package was refactored to use a configure file (yaml) to parse parameters. In addition, ocker container images
are provided with all the necessary dependencies pre-installed. Various models were tested for consistency
(others were disabled for the time being) with propagation delays computation support for zenith, and slant through
ray-tracing using orbit file. Modules were restructured for computational performance improvement.

This release is the initial release to the NISAR ADT for supporting the generation of the stratospheric ancillary
correction using HRES model.

### New/Updated Features
+ Supported for models (verified)
   - ECWMF: HRES, ERA5, and ERA5T access verified and tested
   - NASA GSFC: GMAO access fix for PYDAP changes
   - NOAA: HRRR access through AWS S3 bucket (using [Herbie](https://github.com/blaylockbk/Herbie)) and correction on
     incorrect loading of the pressure.
   - NCMR: not tested
   - Other models are currently disabled (MERRA-2, WRF, ERA-I)
+ Two flavors for computing propagation delays
   - Zenith delays
   - Slant delays through ray-tracing (supported through orbit file)
+ Refactoring and computational improvements for delay computation
   - Better organized for individual function calls
+ When using a pre-existing weather model file, raise a warning instead of an error when the existing weather model
  doesn't cover the entire AOI.
+ Support for delay outputs at
   - 2D coordinate list (x,y) for a user-defined height or at topographic height
   - 2D grid at user-specified output sampling (x,y) and coordinate projection for a user-defined height or by default
     at topographic height
   - 3D cube at weather model grid notes or at user-specified output sampling (x,y,z) and coordinate projection
+  Docker container images are provided with all the necessary dependencies pre-installed. See: <https://github.com/dbekaert/RAiDER/#using-the-docker-image>
+ CLI has changed from `raiderDelay.py` to `raider.py` with options provided through configure file (yaml format).
+ Unit testing for slant ray-tracing
+ RAiDER-Docs documentation updated for changes in RAiDER package

### Not implemented / supported in this release
+ Custom DEMs
+ Pypi (`pip install` capability)
+ Slant delays without an orbit file (e.g. own 2D or 3D LOS files)
+ Conventional slant delays with projection from zenith (e.g. orbit, 2D or 3D LOS files)
+ GUNW product input/output
+ Complete unit test coverage
+ Detailed API documentation
+ Accessing RAiDER as a Python Library.

### Contributors
- Jeremy Maurer
- Brett Buzzanga
- Piyush Agram - _(Descartes Labs)_
- Joseph Kennedy
- Matthew Licari
- David Bekaert

## [0.1.0]

RAiDER release for initial conda channel

### Updates and highlights

+ Zenith and Conventional slant delays on 3D cube
+ intersect with topography to get raster delays
+ initial conda release
+ Bug fixes
+ dem-stitcher for handling dem conversion to ellipsoidal heights

### Contributors

David Bekaert
Jeremy Maurer
Nate Kean

## [0.0.2]

### RAiDER pre-release in support of NISAR troposphere working group
This release includes features in support of the NISAR tropospheric working group recommendations in June 2021. Salient
features include full capabilities for statistical analysis of GNSS ZTD and comparison with weather model ZTD, addition
of the NCMR weather model from ISRO, and unit test suite update.

### Pre-release
- GMAO, MERRA-2, ECMWF - ERA-5, ERA-5T, HRES, HRRR, and NCMR all working models
- Fixed height levels for weather model calculations
- Update tools for statistical analysis and plotting of weather model and GNSS zenith delays
- Unit test suite additions
- Bug fixes

### Team contributions:
David Bekaert
Yang Lei
Simran Sangha
Jeremy Maurer
Charlie Marshak
Brett Buzzanga

## [0.0.1]

### RAiDER pre-release in support of AGU work
This release includes the features in support of the team's 2020 AGU presentation. Salient features include model
download, GNSS ZTD download, ZTD delay calculation at GNSS locations, statistical analysis of the GNSS and model ZTD.
Slant delay and large scale processing in ongoing development.

### Pre-release
- Concurrent download option (tested for GMAO, HRES, ERA5)
- Include download and reader support for historic GMAO (2014-2-20 to 2017-12-01), not support in OpenDap (post 2017-12-01).
- Add support for historic HRES data with 91 model levels (pre-2013-6-26)
- Tools to merge RAiDER GNSS delay files, intersection with GNSS downloaded delays, residual computation
- Ability to re-plot grid analysis on statsPlot without need to re-run analysis.
- Clean up of stats class plotting
- Bug fixes

### Team contributions
Jeremy Maurer
Yang Lei
Simran Sangha
David Bekaert
Brett Buzzanga


## [0.0.0]

### First RAiDER pre-release
Predominant to be used for model download in support of upcoming statistical analysis for NISAR tropospheric noise working group.
Delay calculation and stats class in ongoing development.

### Pre-release
- Download support for HRRR, GMAO, HRES, ERA5, ERAT, MERRA2
- compatible readers for HRRR, GMAO, HRES, ERA5, ERAT, MERRA2, NCUM
- Zenith delay support and draft version for ray-tracing
- support for radar-coordinates, geo-coordinates, native model, and station nodes
- Stats class with GNSS download from UNR
- Stats class for performance evaluation (variogram, correlation, std, bias) with plotting
- Model documentation and first cut of sample Jupyter notebooks

### Team contributions
Jeremy Maurer
Raymond Hogenson
Yang Lei
Rohan Weeden
Simran Sangha
Heresh Fattahi
David Bekaert

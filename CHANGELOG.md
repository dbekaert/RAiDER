# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3]
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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/) 
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

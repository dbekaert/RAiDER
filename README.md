# RAiDER
Raytracing Atmospheric Delay Estimation for RADAR

[![Language](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/dbekaert/RAiDER/blob/dev/LICENSE)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3a787083f97646e1856efefab69374a8)](https://www.codacy.com/manual/bekaertdavid/RAiDER?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dbekaert/RAiDER&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/dbekaert/RAiDER.svg?style=svg)](https://circleci.com/gh/dbekaert/RAiDER)
[![Coverage Status](https://coveralls.io/repos/github/dbekaert/RAiDER/badge.svg?branch=dev)](https://coveralls.io/github/dbekaert/RAiDER?branch=dev)

RAiDER-tools is a package in Python which contains tools to calculate tropospheric corrections for Radar using a raytracing implementation. Its development was funded under the NASA Sea-level Change Team (NSLCT) program, the Earth Surface and Interior (ESI) program, and the NISAR Science Team (NISAR-ST) (NTR-51433). U.S. Government sponsorship acknowledged.

Copyright (c) 2019-2021, California Institute of Technology ("Caltech"). All rights reserved.  

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Contents

1. [Software Dependencies](#package-dependencies)
2. [Installation](Installation.md)
3. [Setup for weather model access](WeatherModels.md)
4. [Running RAiDER](#quick-start-with-conda)
5. [Documentation](#documentation)
6. [Citation](#citation)
7. [Contributors and community contributions](#contributors)


------

## Package Dependencies
A complete list of dependencies for RAiDER are listed in the environment.yml file. 

### Python dependencies

* [Python](https://www.python.org/) >= 3  (>= 3.7 preferred)
* [cdsapi](https://pypi.org/project/cdsapi/)
* [cfgrib](https://pypi.org/project/cfgrib/)
* [cmake](https://cmake.org/)
* [cython](https://cython.org/)
* [gdal](https://www.gdal.org/) >= 3.0.0
* [h5py](https://pypi.org/project/h5py/)
* [netcdf4](http://unidata.github.io/netcdf4-python/netCDF4/index.html)
* [numpy](https://numpy.org/)
* [pygrib](https://jswhit.github.io/pygrib/docs/)
* [pandas](https://pandas.pydata.org/)
* [pydap](https://www.pydap.org/en/latest/#) >= 3.2.3
* [pyproj](https://pypi.org/project/pyproj/) >=2.1.0
* [SciPy](https://www.scipy.org/)
* [xarray](http://xarray.pydata.org/en/stable/)

### Python Jupyter dependencies
For the best experience using RAiDER with Jupyter, see [Installing jupyter_contrib_nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) webpage. 

------
## Quick-Start with Conda

```
git clone https://github.com/dbekaert/RAiDER.git
conda env create -f ./RAiDER/environment.yml
conda activate RAiDER
python setup.py install
py.test test/
raiderDelay.py -h 
```

------
## Documentation

For detailed documentation, examples, and Jupyter notebooks see the [RAiDER-docs repository](https://github.com/dbekaert/RAiDER-docs). 
We welcome contributions of other examples on how to leverage the RAiDER  (see [here](https://github.com/dbekaert/RAiDER/blob/master/CONTRIBUTING.md) for instructions).
``` raiderDelay.py -h ``` provides a help menu and list of example commands to get started. 
The RAiDER scripts are highly modulized in Python and allows for building your own processing workflow. 

------

## Citation
TODO

------
## Contributors
* David Bekaert
* Jeremy Maurer
* Raymond Hogenson
* Heresh Fattahi
* Yang Lei
* Rohan Weeden
* Simran Sangha
* [_other community members_](https://github.com/dbekaert/RAiDER/graphs/contributors)

We welcome community contributions. For instructions see [here](https://github.com/dbekaert/RAiDER/blob/dev/CONTRIBUTING.md).

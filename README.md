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

- [RAiDER](#raider)
  - [Contents](#contents)
  - [1. Software Dependencies](#1-software-dependencies)
    - [Python dependencies](#python-dependencies)
    - [Python Jupyter dependencies](#python-jupyter-dependencies)
  - [2. Downloading RAiDER](#2-downloading-raider)
  - [3. Installing RAiDER](#3-installing-raider)
  - [With Conda](#with-conda)
  - [Other Installation Options](#other-installation-options)
  - [Common Installation Issues](#common-installation-issues)
  - [Testing your installation](#testing-your-installation)
  - [4. Setup of third party weather model access](#4-setup-of-third-party-weather-model-access)
  - [5. Running RAiDER and Documentation](#5-running-raider-and-documentation)
  - [6. Citation](#6-citation)
  - [7. Contributors](#7-contributors)


------

## 1. Software Dependencies
Below we list the dependencies for RAiDER.
A complete list is also provided in the environment.yml file.

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
* [rasterio](https://rasterio.readthedocs.io/en/latest/) >=1.2.1
* [SciPy](https://www.scipy.org/)
* [xarray](http://xarray.pydata.org/en/stable/)

### Python Jupyter dependencies
For the best experience using RAiDER with Jupyter, see [Installing jupyter_contrib_nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) webpage.

------
## 2. Downloading RAiDER

Option 1: __[download the source code](https://github.com/dbekaert/RAiDER/archive/dev.zip)__ for RAiDER and unzip to the location where you want to keep the code

Option 2: __[clone to the repository](https://github.com/dbekaert/RAiDER)__ to your system.
```
git clone https://github.com/dbekaert/RAiDER.git
```

------
## 3. Installing RAiDER

RAiDER currently works on \*nix systems, and has been tested on the following systems:
- Ubuntu v.16 and up
- Mac OS v.10 and up

## With Conda
RAiDER was designed to work with __[Conda](https://docs.conda.io/en/latest/index.html)__ a cross-platform way to use Python that allows you to setup and use "virtual environments." These can help to keep dependencies for different sets of code separate. Conda is distrubed as __[Anaconda](https://www.anaconda.com/products/individual)__ or __[Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)__, a light-weight version of Anaconda. See __[here](https://docs.anaconda.com/anaconda/install/)__ for help installing Anaconda and __[here](https://docs.conda.io/en/latest/miniconda.html)__ for installing Miniconda.

```
git clone https://github.com/dbekaert/RAiDER.git
conda env create -f ./RAiDER/environment.yml
conda activate RAiDER
python setup.py install

```

## Other Installation Options
If not using the setup.py, users should ensure RAiDER and dependencies are included on their PATH and PYTHONPATH, and the Geometry module is compiled such it can be imported as Raider.Geo2rdr. For c-shell this can be done as follows (replace "RAiDERREPO" to the location where you have cloned the RAiDER repository):
```
setenv PYTHONPATH $PYTHONPATH:/RAiDERREPO/tools/RAiDER
set PATH $PATH:'/RAiDERREPO/tools/bin'
```

## Common Installation Issues

1. This package uses GDAL and g++, both of which can be tricky to set up correctly.
GDAL in particular will often break after installing a new program
If you receive error messages such as the following:

```
ImportError: ~/anaconda3/envs/RAiDER/lib/python3.7/site-packages/matplotlib/../../../libstdc++.so.6: version `CXXABI_1.3.9' not found (required by ~/anaconda3/envs/RAiDER/lib/python3.7/site-packages/matplotlib/ft2font.cpython-37m-x86_64-linux-gnu.so)
ImportError: libtiledb.so.1.6.0: cannot open shared object file: No such file or directory
***cmake: ~/anaconda3/envs/RAiDER/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by cmake)***
```

try running the following commands within your RAiDER conda environment:
```
conda update --force-reinstall libstdcxx-ng
conda update --force-reinstall gdal libgdal
```

2. This package requires both C++ and C headers, and the system headers are used for some C libraries. If running on a Mac computer, and "python setup.py build" results in a message stating that some system library header file is missing, try the following steps, and accept the various licenses and step through the installation process. Try re-running the build step after each update:

 ```
 xcode-select --install
 open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
 ```

## Testing your installation
To test the installation was successfull you can run the following tests:
```
py.test test/
raiderDelay.py -h
```

------
## 4. Setup of third party weather model access
RAiDER has the ability to download weather models from third-parties; some of which require license agreements. See [here](WeatherModels.md) for details.

------
## 5. Running RAiDER and Documentation
For detailed documentation, examples, and Jupyter notebooks see the [RAiDER-docs repository](https://github.com/dbekaert/RAiDER-docs).
We welcome contributions of other examples on how to leverage the RAiDER  (see [here](https://github.com/dbekaert/RAiDER/blob/master/CONTRIBUTING.md) for instructions).
``` raiderDelay.py -h ``` provides a help menu and list of example commands to get started.
The RAiDER scripts are highly modulized in Python and allows for building your own processing workflow.

------
## 6. Citation
TODO

------
## 7. Contributors
* David Bekaert
* Jeremy Mauarer
* Raymond Hogenson
* Heresh Fattahi
* Yang Lei
* Rohan Weeden
* Simran Sangha
* [_other community members_](https://github.com/dbekaert/RAiDER/graphs/contributors)

We welcome community contributions. For instructions see [here](https://github.com/dbekaert/RAiDER/blob/dev/CONTRIBUTING.md).

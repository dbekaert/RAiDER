# RAiDER
Raytracing Atmospheric Delay Estimation for RADAR

[![Language](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/dbekaert/RAiDER/blob/dev/LICENSE)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/486716578ab549738e3b0485be1b0047)](https://www.codacy.com/manual/bekaertdavid/RAiDER?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dbekaert/RAiDER&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/dbekaert/RAiDER.svg?style=svg)](https://circleci.com/gh/dbekaert/RAiDER)

RAiDER-tools is a package in Python which contains tools to calculate tropospheric corrections for Radar using a raytracing implementation. Its development was funded under the NASA Sea-level Change Team (NSLCT) program and the Earth Surface and Interior (ESI) program (NTR-51433). U.S. Government sponsorship acknowledged. 


Copyright (c) 2019-2020, California Institute of Technology ("Caltech"). All rights reserved.  

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Contents

1. [Software Dependencies](#software-dependencies)
2. [Installation](#installation)
- [Setup for various weather model access](#modelsetup)
- [Other installation options](#installopts)
- [Common Issues](#problems)
3. [Running RAiDER](#running-raider)
4. [Documentation](#documentation)
5. [Citation](#citation)
6. [Contributors and community contributions](#contributors)


------

## Software Dependencies
Below we list the dependencies for RAiDER

### Packages:
```
* Python >= 3  (>= 3.7 preferred)
* [GDAL](https://www.gdal.org/), lib-gdal and its Python bindings >= 3.0
* [cmake](https://cmake.org/)
```

### Python dependencies
```
* [SciPy](https://www.scipy.org/)
* [netcdf4](http://unidata.github.io/netcdf4-python/netCDF4/index.html)
* [cdsapi](https://pypi.org/project/cdsapi/)
* [cfgrib](https://pypi.org/project/cfgrib/)
* [pygrib](https://jswhit.github.io/pygrib/docs/)
* [cython](https://cython.org/)
* [pyproj](https://pypi.org/project/pyproj/) >=2.1.0
* [h5py](https://pypi.org/project/h5py/)
* numpy
* pandas
* xarray
* pydap 3.2.1
```


### Python Jupyter dependencies
```
* py3X-jupyter
* py3X-jupyter_client
* py3X-jupyter_contrib_nbextensions
* py3X-jupyter_nbextensions_configurator
* py3X-hide_code
* py3X-RISE
```


------
## Installation
RAiDER package can be easily installed and used after the dependencies are installed and activated.  Easiest way of installing RAiDER is to use the setup.py script as outlined below. For the required dependencies, we strongly recommend using [Anaconda](https://www.anaconda.com/distribution/) package manager for easy installation of dependencies in the python environment.

Below we outline the different steps for setting up the RAiDER while leveraging Anaconda for installation of the requirements. Running the commands below will clone the RAiDER package to your local directory, create a conda environment with the name 'RAiDER', install dependencies to this environment and activate it.

```
git clone https://github.com/dbekaert/RAiDER.git
conda env create -f ./RAiDER/environment.yml
conda activate RAiDER
```

We have included a setup.py script which allows for easy compilation and installation of dependencies (c-code), as well as setting up the RAiDER package itself (python and command line tools).
```
python setup.py build
python setup.py install
```

If not using the setup.py, users should ensure RAiDER and dependencies are included on their PATH and PYTHONPATH, and the Geometry module is compiled such it can be imported as Raider.Geo2rdr. For c-shell this can be done as follows (replace "RAiDERREPO" to the location where you have cloned the RAiDER repository):
```
setenv PYTHONPATH $PYTHONPATH:/RAiDERREPO/tools/RAiDER
set PATH $PATH:'/RAiDERREPO/tools/bin'
```

### Setup for various weather model access 
The notes to set up various weather model access can be found [here](./weather_setup.md).

### Other installation options
The following pages might be of use to those trying to build third party packages from source.

### Common Issues 

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
 

------
## Running RAiDER

The RAiDER scripts are highly modulized in Python and therefore allows for building your own processing workflow. Below, we show how to call some of the functionality. For detailed documentation, examples, and Jupyter notebooks see the [RAiDER-docs repository](https://github.com/dbekaert/RAiDER-docs). We welcome contributions of other examples on how to leverage the RAiDER  (see [here](https://github.com/dbekaert/RAiDER/blob/master/CONTRIBUTING.md) for instructions).



------
## Documentation

See the [RAiDER-docs repository](https://github.com/dbekaert/RAiDER-docs) for all documentation and Jupyter Notebook Tutorials.

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
* [_other community members_](https://github.com/dbekaert/RAiDER/graphs/contributors)

We welcome community contributions. For instructions see [here](https://github.com/dbekaert/RAiDER/blob/dev/CONTRIBUTING.md).


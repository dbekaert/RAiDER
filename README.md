# RAiDER
Raytracing Atmospheric Delay Estimation for RADAR

[![Language](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)

RAiDER-tools is a package in Python which contains tools to calculate tropospheric corrections for Radar using a raytracing implementation. Its development was funded under the NASA Sea-level Change Team (NSLCT) program and the Earth Surface and Interior (ESI) program.

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Contents

1. [Software Dependencies](#software-dependencies)
2. [Installation](#installation)
3. [Running RAiDER](#running-raider)
- [dummy](#dummy)
4. [Documentation](#documentation)
5. [Citation](#citation)
6. [Contributors and community contributions](#contributors)


------

## Software Dependencies
Below we list the dependencies for RAiDER

### Packages:
```
* Python >= 3.5  (3.6 preferred)
* [PROJ 4](https://github.com/OSGeo/proj) github) >= 6.0
* [GDAL](https://www.gdal.org/) and its Python bindings >= 3.0
```

### Python dependencies
```
* [SciPy](https://www.scipy.org/)
* [netcdf4](http://unidata.github.io/netcdf4-python/netCDF4/index.html)
* [requests](https://2.python-requests.org/en/master/)
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
git clone https://github-fn.jpl.nasa.gov/InSAR-tools/RAiDER.git
conda env create -f ./RAiDER/environment.yml
conda activate RAiDER
```

We have included a setup.py script which allows for easy compilation and installation of dependencies (c-code), as well as setting up the RAiDER package itself (python and command line tools).
```
python setup.py build
python setup.py install
```

If not using the setup.py, users should ensure RAiDER and dependencies are included on their PATH and PYTHONPATH. For c-shell this can be done as follows (replace "RAiDERREPO" to the location where you have cloned the RAiDER repository):
```
setenv PYTHONPATH $PYTHONPATH:/RAiDERREPO/tools/RAiDER
set PATH $PATH:'/RAiDERREPO/tools/bin'
```


### Other installation options
The following pages might be of use to those trying to build third party packages from source.

------
## Running RAiDER

The RAiDER scripts are highly modulized in Python and therefore allows for building your own processing workflow. Below, we show how to call some of the functionality. For detailed documentation, examples, and Jupyter notebooks see the [RAiDER-docs repository](https://github-fn.jpl.nasa.gov/InSAR-tools/RAiDER-docs). We welcome contributions of other examples on how to leverage the RAiDER  (see [here](https://github-fn.jpl.nasa.gov/InSAR-tools/RAiDER/blob/master/CONTRIBUTING.md) for instructions).

### dummy
EXAMPLE TODO


------
## Documentation

See the [RAiDER-docs repository](https://github-fn.jpl.nasa.gov/InSAR-tools/RAiDER-docs) for all documentation and Jupyter Notebook Tutorials.

------
## Citation
TODO

------
## Contributors    
* David Bekaert
* Jeremy Maurer
* Raymond Hogenson
* [_other community members_](https://github-fn.jpl.nasa.gov/InSAR-tools/RAiDER/graphs/contributors)

We welcome community contributions. For instructions see [here](https://github-fn.jpl.nasa.gov/InSAR-tools/RAiDER/blob/master/CONTRIBUTING.md).




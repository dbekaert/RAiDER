# Installing RAiDER

RAiDER currently works on \*nix systems, and has been tested on the following systems:
- Ubuntu v.16 and up
- Mac OS v.10 and up

## Contents

1. [Downloading](#downloading)
2. [Installation with Conda](#installing-with-anaconda-or-miniconda)
2. [Installation without Conda](#installing-without-conda)

## 1. Downloading 
RAiDER package can be easily installed and used after the dependencies are installed and activated.  Easiest way of installing RAiDER is to use the setup.py script as outlined below. For the required dependencies, we strongly recommend using [Anaconda](https://www.anaconda.com/distribution/) package manager for easy installation of dependencies in the python environment.

Option 1: __[download the source code](https://github.com/dbekaert/RAiDER/archive/dev.zip)__ for RAiDER and unzip to the location where you want to keep the code

Option 2: __[clone to the repository](https://github.com/dbekaert/RAiDER)__ to your system.  

Below we outline the different steps for setting up the RAiDER while leveraging Anaconda for installation of the requirements. Running the commands below will clone the RAiDER package to your local directory, create a conda environment with the name 'RAiDER', install dependencies to this environment and activate it.

## 2. Installing with Anaconda or Miniconda
RAiDER was designed to work with __[Conda](https://docs.conda.io/en/latest/index.html)__ a cross-platform way to use Python that allows you to setup and use "virtual environments." These can help to keep dependencies for different sets of code separate. Conda is distrubed as __[Anaconda](https://www.anaconda.com/products/individual)__ or __[Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)__, a light-weight version of Anaconda. See __[here](https://docs.anaconda.com/anaconda/install/)__ for help installing Anaconda and __[here](https://docs.conda.io/en/latest/miniconda.html)__ for installing Miniconda. 

Steps:
1. On a terminal, change to the source code directory and create a new conda environment using ```conda env create -f environment.yml```
2. Activate the new conda environment using ```conda activate RAiDER```
3. From inside the main source code directory, type ```python setup.py install```
4. Finally, test the new installation: ```py.test test/```

## 3. Installing without Conda
If not using the setup.py, users should ensure RAiDER and dependencies are included on their PATH and PYTHONPATH, and the Geometry module is compiled such it can be imported as Raider.Geo2rdr. For c-shell this can be done as follows (replace "RAiDERREPO" to the location where you have cloned the RAiDER repository):
```
setenv PYTHONPATH $PYTHONPATH:/RAiDERREPO/tools/RAiDER
set PATH $PATH:'/RAiDERREPO/tools/bin'
```

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

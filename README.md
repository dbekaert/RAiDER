# RAiDER
Raytracing Atmospheric Delay Estimation for RADAR

[![Language](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/dbekaert/RAiDER/blob/dev/LICENSE)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3a787083f97646e1856efefab69374a8)](https://www.codacy.com/manual/bekaertdavid/RAiDER?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dbekaert/RAiDER&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/dbekaert/RAiDER.svg?style=svg)](https://circleci.com/gh/dbekaert/RAiDER)
[![Coverage Status](https://coveralls.io/repos/github/dbekaert/RAiDER/badge.svg?branch=dev)](https://coveralls.io/github/dbekaert/RAiDER?branch=dev)

RAiDER-tools is a package in Python which contains tools to calculate tropospheric corrections for Radar using a raytracing implementation. Its development was funded under the NASA Sea-level Change Team (NSLCT) program, the Earth Surface and Interior (ESI) program, and the NISAR Science Team (NISAR-ST) (NTR-51433). U.S. Government sponsorship acknowledged.

Copyright (c) 2019-2022, California Institute of Technology ("Caltech"). All rights reserved.

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Contents

- [RAiDER](#raider)
  - [Contents](#contents)
  - [1. Installing RAiDER](#3-installing-raider)
    - [With Conda](#with-conda)
    - [From Source](#3-installing-raider)
  - [2. Software Dependencies](#1-software-dependencies)
    - [Python dependencies](#python-dependencies)
    - [Python Jupyter dependencies](#python-jupyter-dependencies)
  - [3. Setup of third-party weather model access](#4-setup-of-third-party-weather-model-access)
  - [4. Running RAiDER and Documentation](#5-running-raider-and-documentation)
  - [5. Citing](#6-citation)
  - [6. Contributors](#7-contributors)

------
## 1. Getting Started

RAiDER has been tested on the following systems:
- Ubuntu v.16 and up
- Mac OS v.10 and up
### Installing With Conda
RAiDER is available on [conda-forge](https://anaconda.org/conda-forge/raider). __[Conda](https://docs.conda.io/en/latest/index.html)__ is a cross-platform way to use Python that allows you to setup and use "virtual environments." These can help to keep dependencies for different sets of code separate. We recommend using [Miniforge](https://github.com/conda-forge/miniforge), a conda environment manager that uses conda-forge as its default code repo. Alternatively,see __[here](https://docs.anaconda.com/anaconda/install/)__ for help installing Anaconda and __[here](https://docs.conda.io/en/latest/miniconda.html)__ for installing Miniconda.

Installing RAiDER:
```
conda env create --name RAiDER
conda activate RAiDER
conda install -c conda-forge raider
```
### Installing from source
You can also install RAiDER directly from source. Doing so is recommended for those who would like to [contribute to the source code](https://github.com/dbekaert/RAiDER/blob/dev/CONTRIBUTING.md), which we heartily encourage! For more details on installing from source see [here](https://github.com/dbekaert/RAiDER/blob/dev/Installing_from_source.md).
```
git clone https://github.com/dbekaert/RAiDER.git
cd RAiDER
conda env create -f environment.yml
# can also use the development version: 
# conda env create -f environment-dev.yml
conda activate RAiDER
pip install -e .
```
------
## 3. Setup of third party weather model access
RAiDER has the ability to download weather models from third-parties; some of which require license agreements. See [here](WeatherModels.md) for details.

------
## 4. Running RAiDER and Documentation
For detailed documentation, examples, and Jupyter notebooks see the [RAiDER-docs repository](https://github.com/dbekaert/RAiDER-docs).
We welcome contributions of other examples on how to leverage the RAiDER  (see [here](https://github.com/dbekaert/RAiDER/blob/master/CONTRIBUTING.md) for instructions).
``` raiderDelay.py -h ``` provides a help menu and list of example commands to get started.
The RAiDER scripts are highly modulized in Python and allows for building your own processing workflow.

------
## 5. Citation
TODO

------
## 7. Contributors
* David Bekaert
* Jeremy Mauarer
* Raymond Hogenson
* Yang Lei
* Rohan Weeden
* Simran Sangha
* [_other community members_](https://github.com/dbekaert/RAiDER/graphs/contributors)

We welcome community contributions! For instructions see [here](https://github.com/dbekaert/RAiDER/blob/dev/CONTRIBUTING.md).

#!/usr/bin/env python
# -------------------------------------------------------------------
# - NAME:        ERA5.py
# - AUTHOR:      Reto Stauffer
# - DATE:        2017-01-05
# -------------------------------------------------------------------
# - DESCRIPTION: ERA5 data downloader (currently for the Alps, check
#                lonmin, lonmax, latmin, latmax!
# -------------------------------------------------------------------
# - EDITORIAL:   2017-01-05, RS: Created file on pc24-c707.
# -------------------------------------------------------------------
# - L@ST MODIFIED: 2018-12-15 15:48 on marvin
# -------------------------------------------------------------------

import logging as log
log.basicConfig(format='%(levelname)s: %(message)s', level = log.DEBUG)

# -------------------------------------------------------------------
# Main script part
# -------------------------------------------------------------------
if __name__ == "__main__":

    # Using sys.args to control the date which should be processed.
    from optparse import OptionParser
    import sys
    usage = """Usage:

    {0:s} --years <years> --parameter <parameter> --level <level>
    {0:s} -y <years> -p <parameter> -l <level>

    Downloading ERA5 reanalysis data from Copernicus Climate Data
    Services. NOTE: Not yet made for ensemble ERA5!

    Requires `cdsapi` to be installed (python package to access
    Copernicus Climate Data Services, CDS).

    How to install cdsapi and the required API key:
    * https://cds.climate.copernicus.eu/api-how-to

    Available parameters for "single level" (-t/--type sf):
    * https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

    Available parameters for "pressure level" (-t/--type pl):
    * https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

    Usage examples:

    10m wind speed for the year 2018
    >>> python {0:s} --years 2018 --parameter 10m_u_component_of_wind

    10m wind speed for 2010 and 2018
    >>> python {0:s} --years 2010,2018 --parameter 10m_u_component_of_wind

    10m wind speed for 2008, 2009, 2010, ..., 2018
    >>> python {0:s} --years 2008-2018 --parameter 10m_u_component_of_wind

    700hPa geopotential height, 2018
    >>> python {0:s} --years 2018 --parameter geopotential --level 700
    
    """.format(__file__)
    parser = OptionParser(usage = usage)
    parser.add_option("-y", "--years", dest = "years", type = "str",
            help="years for which the data should be downloaded. " + \
                 "Can either be a single year (e.g., 2010), a comma separated list " + \
                 "(e.g., 2010,2011), or a sequence of the form <bgn>-<end> " + \
                 "(e.g., 2010-2018; 2010 to 2018).")
    parser.add_option("-p", "--parameter", dest = "param", type = "str",
            help="the parameter/variable to be downloaded. Please " + \
                 "check the cds website to see what's availabe.")
    parser.add_option("-l", "--level", dest = "level", type = "int", default = None,
            help="level, only for pressure level data. For \"single level data\" " + \
                 "(e.g., surface level variables) do not set this parameter!")
    parser.add_option("-t", "--test", dest = "test", default = False, action = "store_true",
            help="development flag. If set, only one day (January 1th of the first " + \
                 "year) will be downloaded before the script stops. " + \
                 "The output file will be labeled with \"_test_\".")
    (options,args) = parser.parse_args()


    # Subsetting, currently hardcoded/fixed
    lonmin =  2.0
    lonmax = 20.0
    latmin = 40.0
    latmax = 52.0
    the_subset = "/".join(["{:.2f}".format(x) for x in [latmax, lonmin, latmin, lonmax]])

    # Output directory
    datadir = "era5_data"

    # Missing date?
    import re
    from numpy import min, max, arange, unique, sort
    if options.years == None or options.param == None:
        parser.print_help(); sys.exit(9)
    else:
        if re.match("^[0-9]{4}$", options.years):
            options.years = [int(options.years)]
        elif re.match("^[0-9,]+$", options.years):
            options.years = [int(x) for x in options.years.split(",")]
        elif re.match("^[0-9]{4}-[0-9]{4}$", options.years):
            options.years = [int(x) for x in options.years.split("-")]
            options.years = list(arange(min(options.years), max(options.years)+1))
        else:
            parser.print_help(); log.info("\n\n")
            raise ValueError("Wrong format for -y/--years")

        # Sort and unique 
        options.years = list(sort(unique(options.years)))
        options.years.reverse()

    # Minimum year
    if min(options.years) < 1950:
        raise ValueError("input years is definitively wrong as before 1950!")

    # Quick output
    log.info("\n{:s}".format("".join(["-"]*60)))
    log.info("Processing data for years: {:s}".format(", ".join([str(x) for x in options.years])))
    log.info("Parameter: {:s}".format(options.param))
    log.info("Subset: {:s}\n".format(the_subset))
    if options.level is None:
        log.info("Single level parameter")
    else:
        log.info("Vertical level: {:d} hPa".format(options.level))
    if options.test:
        log.info("TEST MODE ACTIVE: Only one day will be downloaded!")
    log.info("{:s}\n".format("".join(["-"]*60)))

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    import sys, os
    import datetime as dt
    import subprocess as sub
    from cdsapi import Client
    from datetime import date
    from numpy import arange

    # Output directory
    tmp = "" if options.level is None else "_{:d}hPa".format(options.level)
    paramdir = os.path.join(datadir, "{:s}{:s}".format(options.param,tmp))
    if not os.path.isdir(paramdir):
        try:
            os.makedirs(paramdir)
        except:
            raise Exception("not able to create directory \"{;s}\"".format(datadir))

    # range(2017,2018) downloads data for 2017
    for year in options.years:

        if year > int(date.today().strftime("%Y")): continue
        log.info("[!] Processing year {:04d}".format(year))

        # ----------------------------------------------------------------
        # SURFACE DATA
        # ----------------------------------------------------------------
        tmp     = "" if options.level is None else "_{:d}hPa".format(options.level)
        # File name without suffix
        tmp     = "ERA5_{:04d}{:s}_{:s}{:s}".format(year, \
                  "_test" if options.test else "", options.param, tmp)
        # Create the different file names we need
        ncfile  = os.path.join(paramdir, "{:s}.nc".format(tmp))

        # If netcdf file exists: skip
        if os.path.isfile(ncfile):
           log.info("Output file \"{:s}\" exists, proceed with next ...".format(ncfile))
        else:

           # Request
           log.info("Downloading: {:s}".format(ncfile))
           args = {"product_type" : "reanalysis",
                   "format"       : "netcdf",
                   "area"         : the_subset,
                   "variable"     : options.param,
                   "year"         : "{:04d}".format(year),
                   "month"        : ["{:02d}".format(x) for x in range(1,13)],
                   "day"          : ["{:02d}".format(x) for x in range(1,32)],
                   "time"         : ["{:02d}:00".format(x) for x in range(0,24)]}
           # Level
           if not options.level == None:
               args["pressure_level"] = "{:d}".format(options.level)

           # Test mode?
           if options.test:
               args["day"]   = "01"
               args["month"] = "01"

           for key,val in args.items():
               if isinstance(val, str):
                   log.info("{:20s} {:s}".format(key, val))
               else:
                   log.info("{:20s} {:s}".format(key, ", ".join(list(val))[0:40]))

           try:
               server = Client()
               if options.level:
                  cdstype = "reanalysis-era5-pressure-levels"
               else:
                  cdstype = "reanalysis-era5-single-levels"
               # Downloading data
               server.retrieve(cdstype, args, ncfile)
               log.info("[+] Everything seems to be fine for {0}".format(ncfile))

           except Exception as e:
               log.info(e)
               log.info("[!] PROBLEMS DETECTED")
               log.info("    cdsapi returned error code != 0")
               log.info("    for file: {0}".format(ncfile))
               if options.test: log.info("\nTEST MODE: Stop here.\n"); sys.exit(0)
               continue

           if options.test: log.info("\nTEST MODE: Stop here.\n"); sys.exit(0)

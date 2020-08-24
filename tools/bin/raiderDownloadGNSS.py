#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran Sangha, Jeremy Maurer, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from RAiDER.downloadGNSSDelays import cmd_line_parse, query_repos

if __name__ == "__main__":
    inps = cmd_line_parse()

    query_repos(
        inps.station_file,
        inps.bounding_box,
        inps.out,
        inps.years,
        inps.returnTime,
        inps.download,
        inps.cpus,
        inps.verbose
    )

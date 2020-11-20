#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran Sangha, Jeremy Maurer, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from RAiDER.statsPlot import cmd_line_parse, stats_analyses

if __name__ == "__main__":
    inps = cmd_line_parse()

    stats_analyses(
        inps.fname,
        inps.col_name,
        inps.unit,
        inps.workdir,
        inps.cpus,
        inps.verbose,
        inps.bounding_box,
        inps.spacing,
        inps.timeinterval,
        inps.seasonalinterval,
        inps.figdpi,
        inps.user_title,
        inps.plot_fmt,
        inps.cbounds,
        inps.colorpercentile,
        inps.densitythreshold,
        inps.stationsongrids,
        inps.drawgridlines,
        inps.plotall,
        inps.station_distribution,
        inps.station_delay_mean,
        inps.station_delay_stdev,
        inps.grid_heatmap,
        inps.grid_delay_mean,
        inps.grid_delay_median,
        inps.grid_delay_stdev,
        inps.grid_delay_absolute_mean,
        inps.grid_delay_absolute_median,
        inps.grid_delay_absolute_stdev,
        inps.grid_to_raster,
        inps.variogramplot,
        inps.binnedvariogram,
        inps.variogram_per_timeslice
    )

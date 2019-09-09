from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import matplotlib as mpl

# supress matplotlib postscript warnings
mpl._log.setLevel('ERROR')


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FixedLocator
from pandas.plotting import register_matplotlib_converters
import warnings
register_matplotlib_converters()
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def makeBasemap(lats, lons, hgts = None, plot_bbox = None, plotName = 'stationDistribution.pdf'):
    ########################Plot station distribution
    basemap   = cimgt.Stamen('terrain-background')

    if plot_bbox is None:
        plot_bbox = (np.nanmin(lons), np.nanmax(lons), np.nanmin(lats), np.nanmax(lats)) 

    #######The plot extent corresponds to image corners###########
    fig, axes = plt.subplots(subplot_kw={'projection':basemap.crs})
    axes.set_extent(plot_bbox, ccrs.Geodetic())
    axes.add_image(basemap, 10)
    axes.coastlines()
    cmap = plt.cm.Greys_r
    cmap.set_under('black')
    cmap.set_bad('black', 0.)
    axes.set_xlabel('longitude',weight='bold', zorder=2)
    axes.set_ylabel('latitude',weight='bold', zorder=2)

    # plot the data
    if hgts is not None:
        im = plt.scatter(lons, lats, c=hgts, s = 8, zorder=1)
    else:
        im = plt.scatter(lons, lats, c='k', s = 8, zorder=1)
    
    cbar_ax=fig.colorbar(im,pad=0.1)
    #cbar_ax.set_label('Station distribution', rotation=-90, labelpad=10)

    # draw gridlines
    axes.set_xticks(np.linspace(plot_bbox[0], plot_bbox[1], 5), crs=ccrs.PlateCarree())
    axes.set_yticks(np.linspace(plot_bbox[2], plot_bbox[3], 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='')
    lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='')
    axes.xaxis.set_major_formatter(lon_formatter)
    axes.yaxis.set_major_formatter(lat_formatter)

    plt.tight_layout()

    plt.savefig(plotName)



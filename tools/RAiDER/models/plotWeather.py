"""
This set of functions is designed to for plotting WeatherModel
class objects. It is not designed to be used on its own apart
from this class.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal

import RAiDER.interpolator as intrp


def plot_pqt(weatherObj, savefig = True, z1 = 500, z2 = 15000):
    '''
    Create a plot with pressure, temp, and humidity at two heights
    '''

    # Get the interpolator
    intFcn= intrp.Interpolator()
    intFcn.setPoints(*weatherObj.getPoints())
    intFcn.setProjection(weatherObj.getProjection())
    intFcn.getInterpFcns(weatherObj._p, weatherObj._e, weatherObj._t, interType = 'scipy')

    # get the points needed
    x = weatherObj._xs[:,:,0]
    y = weatherObj._ys[:,:,0]
    z1a = np.zeros(x.shape) + z1
    z2a = np.zeros(x.shape) + z2
    pts1 = np.stack((x.flatten(), y.flatten(), z1a.flatten()), axis = 1)
    pts2 = np.stack((x.flatten(), y.flatten(), z2a.flatten()), axis = 1)

    p1, e1, t1 = intFcn(pts1)
    p2, e2, t2 = intFcn(pts2)

    # Now get the data to plot
    plots = [p1/1e2, e1/1e2, t1 - 273.15, p2/1e2, e2/1e2, t2 - 273.15]
    titles = ('P (hPa)', 'E (hPa)'.format(z1), 'T (C)', '', '','')

    # setup the plot
    f = plt.figure(figsize = (10,6))
    xind = int(np.floor(weatherObj._xs.shape[0]/2))
    yind = int(np.floor(weatherObj._ys.shape[1]/2))
    # loop over each plot
    for ind, plot, title in zip(range(len(plots)), plots, titles):
        sp = f.add_subplot(3,3,ind + 1)
        if ind != 2:
            sp.xaxis.set_ticklabels([])
            sp.yaxis.set_ticklabels([])
        im = sp.imshow(np.reshape(plot, x.shape),
                       cmap='viridis',
                       extent=(np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)),
                       origin='lower')
        sp.plot(x[xind, yind],y[xind, yind], 'ko', 'filled')
        divider = mal(sp)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax)
        sp.set_title(title)
        if ind==0:
            sp.set_ylabel('{} m'.format(z1))
        if ind==3:
            sp.set_ylabel('{} m'.format(z2))

    # add plots that show each variable with height
    zdata = weatherObj._zs[xind,yind,:]/1000
    sp = f.add_subplot(3,3,7)
    sp.plot(weatherObj._p[xind,yind,:]/1e2, zdata)
    sp.set_ylabel('Height (km)')
    sp.set_xlabel('Pressure (hPa)')

    sp = f.add_subplot(3,3,8)
    sp.plot(weatherObj._e[xind,yind,:]/100,zdata)
    sp.yaxis.set_ticklabels([])
    sp.set_xlabel('E (hPa)')

    sp = f.add_subplot(3,3,9)
    sp.plot(weatherObj._t[xind,yind,:]- 273.15, zdata)
    sp.yaxis.set_ticklabels([])
    sp.set_xlabel('Temp (C)')

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.05,
                    wspace=0.3)

    if savefig:
         plt.savefig('Weather_hgt{}_and_{}m.pdf'.format(z1, z2))
    return f


def plot_wh(weatherObj, savefig = True, z1 = 500, z2 = 15000):
    '''
    Create a plot with wet refractivity and hydrostatic refractivity,
    at two different heights
    '''

    # Get the interpolator
    intFcn= intrp.Interpolator()
    intFcn.setPoints(*weatherObj.getPoints())
    intFcn.setProjection(weatherObj.getProjection())
    intFcn.getInterpFcns(weatherObj._wet_refractivity, weatherObj._hydrostatic_refractivity, interType = 'scipy')

    # get the points needed
    x = weatherObj._xs[:,:,0]
    y = weatherObj._ys[:,:,0]
    z1a = np.zeros(x.shape) + z1
    z2a = np.zeros(x.shape) + z2
    pts1 = np.stack((x.flatten(), y.flatten(), z1a.flatten()), axis = 1)
    pts2 = np.stack((x.flatten(), y.flatten(), z2a.flatten()), axis = 1)

    w1, h1 = intFcn(pts1)
    w2, h2 = intFcn(pts2)

    # Now get the data to plot
    plots = [w1, h1, w2, h2]

    # titles
    titles = ('Wet refractivity ({:5.1f} m)'.format(z1),
              'Hydrostatic refractivity ({:5.1f} m)'.format(z1),
              'Wet refractivity ({:5.1f} m)'.format(z2),
              'Hydrostatic refractivity ({:5.1f} m)'.format(z2))

    # setup the plot
    f = plt.figure(figsize = (6,6))

    # loop over each plot
    for ind, plot, title in zip(range(len(plots)), plots, titles):
        sp = f.add_subplot(2,2,ind + 1)
        sp.xaxis.set_ticklabels([])
        sp.yaxis.set_ticklabels([])
        im = sp.imshow(np.reshape(plot, x.shape), cmap='viridis')
        divider = mal(sp)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax)
        sp.set_title(title)

    if savefig:
         plt.savefig('Refractivity_hgt{}_and_{}m.pdf'.format(z1, z2))
    return f

"""
This set of functions is designed to for plotting WeatherModel
class objects. It is not designed to be used on its own apart
from this class.
"""

import os
from RAiDER.interpolator import RegularGridInterpolator as Interpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def plot_pqt(weatherObj, savefig=True, z1=500, z2=15000):
    '''
    Create a plot with pressure, temp, and humidity at two heights
    '''

    # Get the interpolator

    intFcn_p = Interpolator((weatherObj._xs, weatherObj._ys, weatherObj._zs), weatherObj._p.swapaxes(0, 1))
    intFcn_e = Interpolator((weatherObj._xs, weatherObj._ys, weatherObj._zs), weatherObj._e.swapaxes(0, 1))
    intFcn_t = Interpolator((weatherObj._xs, weatherObj._ys, weatherObj._zs), weatherObj._t.swapaxes(0, 1))

    # get the points needed
    XY = np.meshgrid(weatherObj._xs, weatherObj._ys)
    x = XY[0]
    y = XY[1]
    z1a = np.zeros(x.shape) + z1
    z2a = np.zeros(x.shape) + z2
    pts1 = np.stack((x.flatten(), y.flatten(), z1a.flatten()), axis=1)
    pts2 = np.stack((x.flatten(), y.flatten(), z2a.flatten()), axis=1)

    p1 = intFcn_p(pts1)
    e1 = intFcn_e(pts1)
    t1 = intFcn_t(pts1)
    p2 = intFcn_p(pts2)
    e2 = intFcn_e(pts2)
    t2 = intFcn_t(pts2)

    # Now get the data to plot
    plots = [p1 / 1e2, e1 / 1e2, t1 - 273.15, p2 / 1e2, e2 / 1e2, t2 - 273.15]
    # titles = ('P (hPa)', 'E (hPa)'.format(z1), 'T (C)', '', '', '')
    titles = ('P (hPa)', 'E (hPa)', 'T (C)', '', '', '')

    # setup the plot
    f = plt.figure(figsize=(18, 14))
    f.suptitle('{} Pressure/Humidity/Temperature at height {}m and {}m (values should drop as elevation increases)'.format(weatherObj._Name, z1, z2))

    xind = int(np.floor(weatherObj._xs.shape[0] / 2))
    yind = int(np.floor(weatherObj._ys.shape[0] / 2))

    # loop over each plot
    for ind, plot, title in zip(range(len(plots)), plots, titles):
        sp = f.add_subplot(3, 3, ind + 1)
        im = sp.imshow(np.reshape(plot, x.shape),
                       cmap='viridis',
                       extent=[np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)],
                       origin='lower')
        sp.plot(x[yind, xind], y[yind, xind], 'ko')
        divider = mal(sp)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax)
        sp.set_title(title)
        if ind == 0:
            sp.set_ylabel('{} m\n'.format(z1))
        if ind == 3:
            sp.set_ylabel('{} m\n'.format(z2))

    # add plots that show each variable with height
    zdata = weatherObj._zs[:] / 1000
    sp = f.add_subplot(3, 3, 7)
    sp.plot(weatherObj._p[yind, xind, :] / 1e2, zdata)
    sp.set_ylabel('Height (km)')
    sp.set_xlabel('Pressure (hPa)')

    sp = f.add_subplot(3, 3, 8)
    sp.plot(weatherObj._e[yind, xind, :] / 100, zdata)
    sp.set_xlabel('E (hPa)')

    sp = f.add_subplot(3, 3, 9)
    sp.plot(weatherObj._t[yind, xind, :] - 273.15, zdata)
    sp.set_xlabel('Temp (C)')

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.2,
                        wspace=0.3)

    if savefig:
        wd = os.path.dirname(os.path.dirname(weatherObj.files[0]))
        f  = f'{weatherObj._Name}_weather_hgt{z1}_and_{z2}m.pdf'
        plt.savefig(os.path.join(wd, f))
    return f


def plot_wh(weatherObj, savefig=True, z1=500, z2=15000):
    '''
    Create a plot with wet refractivity and hydrostatic refractivity,
    at two different heights
    '''

    # Get the interpolator
    intFcn_w = Interpolator((weatherObj._xs, weatherObj._ys, weatherObj._zs), weatherObj._wet_refractivity.swapaxes(0, 1))
    intFcn_h = Interpolator((weatherObj._xs, weatherObj._ys, weatherObj._zs), weatherObj._hydrostatic_refractivity.swapaxes(0, 1))

    # get the points needed
    XY = np.meshgrid(weatherObj._xs, weatherObj._ys)
    x = XY[0]
    y = XY[1]
    z1a = np.zeros(x.shape) + z1
    z2a = np.zeros(x.shape) + z2
    pts1 = np.stack((x.flatten(), y.flatten(), z1a.flatten()), axis=1)
    pts2 = np.stack((x.flatten(), y.flatten(), z2a.flatten()), axis=1)

    w1 = intFcn_w(pts1)
    h1 = intFcn_h(pts1)
    w2 = intFcn_w(pts2)
    h2 = intFcn_h(pts2)

    # Now get the data to plot
    plots = [w1, h1, w2, h2]

    # titles
    titles = ('Wet refractivity {}'.format(z1),
              'Hydrostatic refractivity {}'.format(z1),
              '{}'.format(z2),
              '{}'.format(z2))

    # setup the plot
    f = plt.figure(figsize=(14, 10))
    f.suptitle('{} Wet and Hydrostatic refractivity at height {}m and {}m'.format(weatherObj._Name, z1, z2))

    # loop over each plot
    for ind, plot, title in zip(range(len(plots)), plots, titles):
        sp = f.add_subplot(2, 2, ind + 1)
        im = sp.imshow(np.reshape(plot, x.shape), cmap='viridis', extent=[np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)], origin='lower')
        divider = mal(sp)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax)
        sp.set_title(title)
        if ind == 0:
            sp.set_ylabel('{} m\n'.format(z1))
        if ind == 2:
            sp.set_ylabel('{} m\n'.format(z2))

    if savefig:
        plt.savefig('{}_refractivity_hgt{}_and_{}m.pdf'.format(weatherObj._Name, z1, z2))
    return f

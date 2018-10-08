import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import numpy as np
import interpolator as intrp

def plot_pqt(weatherObj, savefig = True, z1 = 500, z2 = 15000):
    '''
    Create a plot with pressure, temp, and humidity at two heights
    '''

    # Get the interpolator 
    intFcn= intrp.Interpolator()
    intFcn.setPoints(*weatherObj.getPoints())
    intFcn.setProjection(weatherObj.getProjection())
    intFcn.getInterpFcns(weatherObj._p, weatherObj._e, weatherObj._t)

    # get the points needed
    x = weatherObj._xs[:,:,0]
    y = weatherObj._ys[:,:,0]
    z1a = np.zeros(x.shape) + z1
    z2a = np.zeros(x.shape) + z2
    pts1 = np.stack((x.flatten(), y.flatten(), z1a.flatten()), axis = 1)
    pts2 = np.stack((x.flatten(), y.flatten(), z2a.flatten()), axis = 1)

    p1, e1, t1 = intFcn(pts1)
    p2, e2, t2 = intFcn(pts2)

    #intFcn.getInterpFcns(weatherObj.getWetRefractivity(), weatherObj.getHydroRefractivity())

    # Now get the data to plot
    plots = [p1/1e5, e1, t1 - 273.15, p2/1e5, e2, t2 - 273.15]

    # titles
    titles = ('Surface P (bars)', 
              'Surface Q ({:5.1f} m)'.format(z1), 
              'Surface T (C)', 
              'High P (bars)', 
              'High Q ({:5.1f} m)'.format(z2), 
              'High T (C)')

    # setup the plot
    f = plt.figure(figsize = (10,6))

    # loop over each plot
    for ind, plot, title in zip(range(len(plots)), plots, titles):
        sp = f.add_subplot(2,3,ind + 1)
        sp.xaxis.set_ticklabels([])
        sp.yaxis.set_ticklabels([])
        im = sp.imshow(np.reshape(plot, x.shape), cmap='viridis')
        divider = mal(sp)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax)
        sp.set_title(title)

        if savefig:
             plt.savefig('Weather_hgt{}_and_{}.pdf'.format(z1, z2))
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
    intFcn.getInterpFcns(weatherObj._wet_refractivity, weatherObj._hydrostatic_refractivity)

    # get the points needed
    x = weatherObj._xs[:,:,0]
    y = weatherObj._ys[:,:,0]
    z1a = np.zeros(x.shape) + z1
    z2a = np.zeros(x.shape) + z2
    pts1 = np.stack((x.flatten(), y.flatten(), z1a.flatten()), axis = 1)
    pts2 = np.stack((x.flatten(), y.flatten(), z2a.flatten()), axis = 1)

    w1, h1 = intFcn(pts1)
    w2, h2 = intFcn(pts2)

    #intFcn.getInterpFcns(weatherObj.getWetRefractivity(), weatherObj.getHydroRefractivity())

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
             plt.savefig('Weather_hgt{}_and_{}.pdf'.format(z1, z2))
        return f

"""Test functions for the ray-tracing suite.

These aren't meant to be automated tests or anything fancy like that,
it's just convenience for when I'm testing everything.
"""


import delay
import netcdf
import numpy
import matplotlib.pyplot as plt


def test_weather():
    """Test the functions with some hard-coded data."""
    return netcdf.load(
            '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                'wrfout_d02_2007-01-30_05:16:00',
            '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                'wrfplev_d02_2007-01-30_05:16:00')


def test_delay(weather):
    """Calculate the delay at a particular place."""
    return delay.dry_delay(weather, 15, -100, -50, delay.Zenith, numpy.inf)


def compare(a_hydro, a_wet, b_hydro, b_wet):
    """Generate a comparison plot."""
    fig = plt.figure()
    def go(img, title=None, vmin=None, vmax=None, ylabel=None):
        a = fig.add_subplot(3, 3, go.count)
        if title:
            a.set_title(title)
        if ylabel:
            plt.ylabel(ylabel)
        plt.imshow(img, vmin=vmin, vmax=vmax)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.colorbar()
        go.count += 1
    go.count = 1
    med_diff = 2.23
    go(a_hydro, 'Hydrostatic', 0, med_diff, 'New')
    go(a_wet, 'Wet', 0, med_diff)
    go(a_hydro + a_wet, 'Combined', 0, med_diff)
    go(b_hydro, vmin=0, vmax=med_diff, ylabel='TRAIN')
    go(b_wet, vmin=0, vmax=med_diff)
    go(b_hydro + b_wet, vmin=0, vmax=med_diff)
    small_diff = 0.03832002288038684
    go(a_hydro - b_hydro, vmin=-small_diff, vmax=small_diff, ylabel='Difference')
    go(a_wet - b_wet, vmin=-small_diff, vmax=small_diff)
    go(a_hydro + a_wet - b_hydro - b_wet, vmin=-small_diff, vmax=small_diff)
    #differences = (a_hydro + a_wet - b_hydro - b_wet).flatten()
    #differences.sort()
    #ninetyfive = differences[int(len(differences) * 0.95)]
    #print('ninetyfive: {}'.format(ninetyfive))
    plt.savefig('comp_train/mexico.pdf')


def nonans(a):
    return a[numpy.logical_not(numpy.isnan(a))]


def ninetyfive(a):
    a = a.copy()
    a.sort()
    return a[int(len(a)*0.05)], a[int(len(a)*0.95)]


saved_weather = None


def generate_plots():
    """Output plots of things compared to TRAIN.

    For testing purposes only.
    """
    # Some easy memoization
    global saved_weather
    if saved_weather is None:
        weather = test_weather()
        saved_weather = weather
    else:
        weather = saved_weather
    hydro, wet = delay.delay_from_files(weather, '/Users/hogenson/lat.rdr', '/Users/hogenson/lon.rdr', '/Users/hogenson/hgt.rdr', parallel=True).T
    hydro = hydro.reshape(-1, 48)
    wet = wet.reshape(-1, 48)
    train_hydro = numpy.load('comp_train/train_hydro.npy')
    train_wet = numpy.load('comp_train/train_wet.npy')
    nc = ninetyfive((hydro + wet - train_hydro - train_wet).flat)
    plt.hist(nonans((hydro + wet - train_hydro - train_wet).flat), range=nc)
    plt.savefig('comp_train/combineddiffhistogram.pdf', bbox_inches='tight')
    plt.clf()
    nh = ninetyfive(nonans((hydro - train_hydro).flat))
    plt.hist(nonans((hydro - train_hydro).flat), range=nh)
    plt.savefig('comp_train/hydrodiffhistogram.pdf', bbox_inches='tight')
    plt.clf()
    nw = ninetyfive(nonans((wet - train_wet).flat))
    plt.hist(nonans((wet - train_wet).flat), range=nw)
    plt.savefig('comp_train/wetdiffhistogram.pdf', bbox_inches='tight')
    plt.clf()
    plt.imshow(hydro + wet - train_hydro - train_wet, vmin=nc[0], vmax=nc[1])
    plt.colorbar()
    plt.savefig('comp_train/combinedmap.pdf', bbox_inches='tight')
    plt.clf()
    plt.imshow(hydro - train_hydro, vmin=nh[0], vmax=nh[1])
    plt.colorbar()
    plt.savefig('comp_train/hydromap.pdf', bbox_inches='tight')
    plt.clf()
    plt.imshow(wet - train_wet, vmin=nw[0], vmax=nw[1])
    plt.colorbar()
    plt.savefig('comp_train/wetmap.pdf', bbox_inches='tight')
    plt.clf()

"""Test functions for the ray-tracing suite.

These aren't meant to be automated tests or anything fancy like that,
it's just convenience for when I'm testing everything.
"""


import delay
import netcdf
import numpy
import pickle
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
    low = a[int(len(a)*0.05)]
    high = a[int(len(a)*0.95)]
    bigger = max(numpy.abs(low), numpy.abs(high))
    return -bigger, bigger


saved_weather = None


def generate_plots(output='pdf', weather=None, hydro=None, wet=None):
    """Output plots of things compared to TRAIN.

    For testing purposes only.
    """
    # Some easy memoization
    if not weather:
        weather = test_weather()
        global saved_weather
        saved_weather = weather
    if hydro is None or wet is None:
        hydro, wet = delay.delay_from_files(weather, '/Users/hogenson/lat.rdr', '/Users/hogenson/lon.rdr', '/Users/hogenson/hgt.rdr', parallel=True).T
        global saved_hydro
        saved_hydro = hydro
        global saved_wet
        saved_wet = wet
    hydro = hydro.reshape(-1, 48)
    wet = wet.reshape(-1, 48)
    train_hydro = numpy.load('comp_train/train_hydro.npy')
    train_wet = numpy.load('comp_train/train_wet.npy')

    def annotate(values):
        plt.annotate(f'mean: {numpy.mean(values):.4f} m\nstandard deviation: {numpy.std(values):.4f} m', xy=(0.05, 0.85), xycoords='axes fraction')

    nc = (-0.05, 0.05)# ninetyfive((hydro + wet - train_hydro - train_wet).flat)
    plt.hist(nonans((hydro + wet - train_hydro - train_wet).flat), range=nc, bins='auto')
    values = nonans((hydro + wet - train_hydro - train_wet).flat)
    annotate(values)
    plt.title('Total')
    plt.savefig(f'comp_train/combineddiffhistogram.{output}', bbox_inches='tight')
    plt.clf()

    nh = (-0.05, 0.05)#ninetyfive(nonans((hydro - train_hydro).flat))
    values = nonans((hydro - train_hydro).flat)
    plt.hist(values, range=nh, bins='auto')
    annotate(values)
    plt.title('Hydrostatic')
    plt.savefig(f'comp_train/hydrodiffhistogram.{output}', bbox_inches='tight')
    plt.clf()

    nw = (-0.05, 0.05)#ninetyfive(nonans((wet - train_wet).flat))
    values = nonans((wet - train_wet).flat)
    plt.hist(values, range=nw, bins='auto')
    annotate(values)
    plt.title('Wet')
    plt.savefig(f'comp_train/wetdiffhistogram.{output}', bbox_inches='tight')
    plt.clf()

    plt.imshow(hydro + wet - train_hydro - train_wet, vmin=nc[0], vmax=nc[1])
    plt.colorbar()
    plt.title('Total')
    plt.savefig(f'comp_train/combinedmap.{output}', bbox_inches='tight')
    plt.clf()

    plt.imshow(hydro - train_hydro, vmin=nh[0], vmax=nh[1])
    plt.colorbar()
    plt.title('Hydrostatic')
    plt.savefig(f'comp_train/hydromap.{output}', bbox_inches='tight')
    plt.clf()

    plt.imshow(wet - train_wet, vmin=nw[0], vmax=nw[1])
    plt.colorbar()
    plt.title('Wet')
    plt.savefig(f'comp_train/wetmap.{output}', bbox_inches='tight')
    plt.clf()


def pickle_dump(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def recalculate_weather():
    weather = test_weather()
    pickle_dump(weather, 'weather')
    return weather


def recalculate_mexico():
    weather = pickle_load('weather')
    hydro, wet = numpy.array(delay.delay_from_files(weather, '/Users/hogenson/lat.rdr', '/Users/hogenson/lon.rdr', '/Users/hogenson/hgt.rdr', parallel=True)).reshape(2,-1,48)
    numpy.save('my_hydro', hydro)
    numpy.save('my_wet', wet)


def compare_with_train():
    with open('weather', 'rb') as f:
        weather = pickle.load(f)
    train_hydro = numpy.load('comp_train/train_hydro.npy')
    train_wet = numpy.load('comp_train/train_wet.npy')
    my_hydro = numpy.load('my_hydro.npy')
    my_wet = numpy.load('my_wet.npy')
    diff = my_hydro + my_wet - train_hydro - train_wet
    plt.imshow(diff, vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.savefig('comp_train/combinedmap.pdf', bbox_inches='tight')

    plt.clf()

    plt.figure(figsize=(4.5,3.5))
    plt.hist(diff.flatten(), range=(-0.05, 0.05), bins='auto')
    plt.savefig('comp_train/combineddiffhistogram.pdf', bbox_inches='tight')

    plt.clf()

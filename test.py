"""Test functions for the ray-tracing suite.

These aren't meant to be automated tests or anything fancy like that,
it's just convenience for when I'm testing everything.
"""


import delay
from osgeo import gdal
import netcdf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy
import util


lat = '/Users/hogenson/lat.rdr'
lon = '/Users/hogenson/lon.rdr'
height = '/Users/hogenson/hgt.rdr'

train_hydro_old = '/Users/hogenson/train-igram/20070802_ZHD.xyz'
train_wet_old= '/Users/hogenson/train-igram/20070802_ZWD.xyz'
train_hydro_new = '/Users/hogenson/igram3/20100810/20100810_ZHD.xyz'
train_wet_new = '/Users/hogenson/igram3/20100810/20100810_ZWD.xyz'
out_old = '/Users/hogenson/igram2/20070802/wrfout_d02_2007-08-02_05:16:00'
plev_old = '/Users/hogenson/igram2/20070802/wrfplev_d02_2007-08-02_05:16:00'
out_new = '/Users/hogenson/igram3/20100810/wrfout_d02_2010-08-10_05:16:00'
plev_new = '/Users/hogenson/igram3/20100810/wrfplev_d02_2010-08-10_05:16:00'

# train_hydro_old = '/Users/hogenson/igram3/20071102/20071102_ZHD.xyz'
# train_wet_old= '/Users/hogenson/igram3/20071102/20071102_ZWD.xyz'
# train_hydro_new = '/Users/hogenson/igram3/20080202/20080202_ZHD.xyz'
# train_wet_new = '/Users/hogenson/igram3/20080202/20080202_ZWD.xyz'
# out_old = '/Users/hogenson/igram3/20071102/wrfout_d02_2007-11-02_05:16:00'
# plev_old = '/Users/hogenson/igram3/20071102/wrfplev_d02_2007-11-02_05:16:00'
# out_new = '/Users/hogenson/igram3/20080202/wrfout_d02_2008-02-02_05:16:00'
# plev_new = '/Users/hogenson/igram3/20080202/wrfplev_d02_2008-02-02_05:16:00'

t_local = '/Users/hogenson/slc/t.npy'
pos_local = '/Users/hogenson/slc/pos.npy'
v_local = '/Users/hogenson/slc/velocity.npy'
t_kriek = '/home/hogenson/t.npy'
pos_kriek = '/home/hogenson/pos.npy'
v_kriek = '/home/hogenson/velocity.npy'


def test_weather(scipy_interpolate=False):
    """Test the functions with some hard-coded data."""
    return netcdf.load(
            '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                'wrfout_d02_2007-01-30_05:16:00',
            '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                'wrfplev_d02_2007-01-30_05:16:00',
            scipy_interpolate=scipy_interpolate)


def test_delay(weather):
    """Calculate the delay at a particular place."""
    return delay.dry_delay(weather, 15, -100, -50, delay.Zenith, np.inf)


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
    return a[np.logical_not(np.isnan(a))]


def ninetyfive(a):
    a = a.copy()
    a.sort()
    low = a[int(len(a)*0.05)]
    high = a[int(len(a)*0.95)]
    bigger = max(np.abs(low), np.abs(high))
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
    train_hydro = np.load('comp_train/train_hydro.npy')
    train_wet = np.load('comp_train/train_wet.npy')

    def annotate(values):
        plt.annotate(f'mean: {np.mean(values):.4f} m\nstandard deviation: {np.std(values):.4f} m', xy=(0.05, 0.85), xycoords='axes fraction')

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
    hydro, wet = np.array(delay.delay_from_files(weather, '/Users/hogenson/lat.rdr', '/Users/hogenson/lon.rdr', '/Users/hogenson/hgt.rdr', parallel=True)).reshape(2,-1,48)
    np.save('my_hydro', hydro)
    np.save('my_wet', wet)


def compare_with_train():
    with open('weather', 'rb') as f:
        weather = pickle.load(f)
    train_hydro = np.load('comp_train/train_hydro.npy')
    train_wet = np.load('comp_train/train_wet.npy')
    my_hydro = np.load('my_hydro.npy')
    my_wet = np.load('my_wet.npy')
    diff = my_hydro + my_wet - train_hydro - train_wet
    plt.imshow(diff, vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.savefig('comp_train/combinedmap.pdf', bbox_inches='tight')

    plt.clf()

    plt.figure(figsize=(4.5,3.5))
    plt.hist(diff.flatten(), range=(-0.05, 0.05), bins='auto')
    plt.savefig('comp_train/combineddiffhistogram.pdf', bbox_inches='tight')

    plt.clf()


def make_plot(out, plev, lat, lon, height, scipy_interpolate=False, los=delay.Zenith):
    weather = netcdf.load(out, plev, scipy_interpolate=scipy_interpolate)
    hydro, wet = delay.delay_from_files(weather, lat, lon, height, parallel=True, los=los)
    return hydro, wet


def make_igram(out1, plev1, out2, plev2, lats, lons, heights,
               los=delay.Zenith):
    hydro1, wet1 = make_plot(out1, plev1, lats, lons, heights)
    hydro2, wet2 = make_plot(out2, plev2, lats, lons, heights)

    # Difference
    hydrodiff = hydro1 - hydro2
    wetdiff = wet1 - wet2

    return hydrodiff + wetdiff


def train_interpolate(hydro, wet, lat, lon):
    lats = util.gdal_open(lat)
    lons = util.gdal_open(lon)

    train_lon, train_lat, train_hydro_raw = np.fromfile(hydro).reshape(-1, 3).T
    _, _, train_wet_raw = np.fromfile(wet).reshape(-1, 3).T

    train_hydro_raw /= 100
    train_wet_raw /= 100

    ok_points = np.logical_and(np.logical_not(np.isnan(train_lat)), np.logical_not(np.isnan(train_lon)))

    train_hydro_inp = scipy.interpolate.LinearNDInterpolator(np.array((train_lat, train_lon)).T[ok_points], train_hydro_raw[ok_points])
    train_wet_inp = scipy.interpolate.LinearNDInterpolator(np.array((train_lat, train_lon)).T[ok_points], train_wet_raw[ok_points])
    train_hydro = train_hydro_inp(np.array((lats.flatten(), lons.flatten())).T)
    train_wet = train_wet_inp(np.array((lats.flatten(), lons.flatten())).T)
    return train_hydro.reshape(lats.shape), train_wet.reshape(lons.shape)


def train_igram(hydro1, wet1, hydro2, wet2, lat, lon):
    train_hydro1, train_wet1 = train_interpolate(hydro1, wet1, lat, lon)
    train_hydro2, train_wet2 = train_interpolate(hydro2, wet2, lat, lon)
    diff = train_hydro1 + train_wet1 - train_hydro2 - train_wet2
    return diff


def state_to_los(t, x, y, z, vx, vy, vz, lon_first, lon_step, lat_first,
                 lat_step, heights):
    global Geo2rdr
    try:
        Geo2rdr
    except NameError:
        import Geo2rdr
    geo2rdr_obj = Geo2rdr.PyGeo2rdr()
    geo2rdr_obj.set_orbit(t, x, y, z, vx, vy, vz)
    geo2rdr_obj.set_geo_coordinate(np.radians(lon_first),
                                   np.radians(lat_first),
                                   np.radians(lon_step), np.radians(lat_step),
                                   heights.astype(np.double,
                                                  casting='same_kind'))
    # compute the radar coordinate for each geo coordinate
    geo2rdr_obj.geo2rdr()

    # get back the line of sight unit vector
    los_x, los_y, los_z = geo2rdr_obj.get_los()


    # get back the slant ranges
    slant_range = geo2rdr_obj.get_slant_range()
    return np.array((los_x, los_y, los_z)) * slant_range


def test_geo2rdr(t_file, pos_file, v_file):
    t = np.load(t_file)
    x, y, z = np.load(pos_file)
    vx, vy, vz = np.load(v_file)
    return state_to_los(t, x, y, z, vx, vy, vz, -99.7, 0.002, 17.99, -0.002,
                        np.zeros((1, 1)))


def run_timeseries(timeseries, prefix, lat, lon, height, los):
    f = h5py.File(timeseries)
    dates = map(lambda x: datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S'), f['dateList'])
    results = None
    for i, date in enumerate(dates):
        out, plev = (os.path.join(prefix, date.strftime(f'wrf{ext}_d01_%Y-%m-%d_%H:%M:%S')) for ext in ('out', 'plev'))
        hydro, wet = make_plot(out, plev, lat, lon, height, los=los)
        if results is None:
            results = np.zeros((len(dates), 2) + hydro.shape)
        results[i][0] = hydro
        results[i][1] = wet
    return results

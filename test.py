"""Test functions for the ray-tracing suite.

These aren't meant to be automated tests or anything fancy like that,
it's just convenience for when I'm testing everything.
"""


import datetime
import delay
from osgeo import gdal
import h5py
import netcdf
import numpy as np
import os
import pickle
import pyproj
import reader
import scipy
import util


lat = '/Users/hogenson/lat.rdr'
lon = '/Users/hogenson/lon.rdr'
height = '/Users/hogenson/hgt.rdr'
los = '/Users/hogenson/los.rdr'

lat_kriek = '/u/k-data/dbekaert/APS_raytracing/Mexico/ALOS/track_188/processing/merged/geom_master_multilooked/lat.rdr'
lon_kriek = '/u/k-data/dbekaert/APS_raytracing/Mexico/ALOS/track_188/processing/merged/geom_master_multilooked/lon.rdr'
height_kriek = '/u/k-data/dbekaert/APS_raytracing/Mexico/ALOS/track_188/processing/merged/geom_master_multilooked/hgt.rdr'
los_kriek = '/u/k-data/dbekaert/APS_raytracing/Mexico/ALOS/track_188/processing/merged/geom_master_multilooked/los.rdr'
timeseries = '/u/k-data/fattahi/Mexico/timeseries_ionoCor.h5'
prefix = '/u/k-data/dbekaert/APS_raytracing/Mexico/ALOS/track_188/WRF'

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
    try:
        return netcdf.load(
                '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                    'wrfout_d02_2007-01-30_05:16:00',
                '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                    'wrfplev_d02_2007-01-30_05:16:00',
                scipy_interpolate=scipy_interpolate)
    except FileNotFoundError:
        return netcdf.load(
                '/u/k-data/dbekaert/APS_raytracing/Mexico/ALOS/track_188/WRF/'
                    '20070130/wrfout_d01_2007-01-30_00:00:00',
                '/u/k-data/dbekaert/APS_raytracing/Mexico/ALOS/track_188/WRF/'
                    '20070130/wrfplev_d01_2007-01-30_00:00:00')


def test_delay(weather):
    """Calculate the delay at a particular place."""
    return delay.dry_delay(weather, 15, -100, -50, delay.Zenith, np.inf)


def compare(a_hydro, a_wet, b_hydro, b_wet):
    """Generate a comparison plot."""
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
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


def make_plot(out, plev, lat, lon, height, scipy_interpolate=False, los=delay.Zenith, raytrace=True):
    weather = netcdf.load(out, plev, scipy_interpolate=scipy_interpolate)
    hydro, wet = delay.delay_from_files(weather, lat, lon, height, parallel=True, los=los, raytrace=raytrace)
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


def test_geo2rdr(t_file, pos_file, v_file):
    t = np.load(t_file)
    x, y, z = np.load(pos_file)
    vx, vy, vz = np.load(v_file)
    return state_to_los(t, x, y, z, vx, vy, vz, -99.7, 0.002, 17.99, -0.002,
                        np.zeros((1, 1)))


def run_timeseries(timeseries, prefix, lat, lon, height, los, raytrace=True):
    with h5py.File(timeseries) as f:
        dates = list(map(lambda x: datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S'), f['dateList']))
        results = None
        for i, date in enumerate(dates):
            out, plev = (os.path.join(prefix, date.strftime(f'%Y%m%d/wrf{ext}_d01_%Y-%m-%d_%H:%M:%S')) for ext in ('out', 'plev'))
            hydro, wet = make_plot(out, plev, lat, lon, height, los=los, raytrace=raytrace)
            if results is None:
                results = np.zeros((len(dates), 2) + hydro.shape)
            results[i][0] = hydro
            results[i][1] = wet
        return results


def make_sim_weather(out, plev):
    outf = scipy.io.netcdf.netcdf_file(out)
    plevf = scipy.io.netcdf.netcdf_file(plev)
    temperature = plevf.variables['T_PL'][0].copy()
    temperature[temperature == -999] = np.nan
    sim_temp = np.broadcast_to(np.mean(temperature, axis=(1, 2)).reshape(-1, 1, 1), temperature.shape)
    humidity = plevf.variables['RH_PL'][0].copy()
    humidity[humidity == -999] = np.nan
    sim_humidity = np.broadcast_to(np.mean(humidity, axis=(1, 2)).reshape(-1, 1, 1), humidity.shape)
    plevs = plevf.variables['P_PL'][0]
    geo_ht = plevf.variables['GHT_PL'][0].copy()
    geo_ht[geo_ht == -999] = np.nan
    sim_geo_ht = np.broadcast_to(np.mean(geo_ht, axis=(1, 2)).reshape(-1, 1, 1), geo_ht.shape)
    lat = outf.variables['XLAT'][0]
    lon = outf.variables['XLONG'][0]

    # Project lat/lon grid so it's regular (for easy interpolation)

    # See http://www.pkrc.net/wrf-lambert.html
    projection = pyproj.Proj(proj='lcc', lat_1=outf.TRUELAT1,
                             lat_2=outf.TRUELAT2, lat_0=outf.MOAD_CEN_LAT,
                             lon_0=outf.STAND_LON, a=6370, b=6370,
                             towgs84=(0,0,0), no_defs=True)

    lla = pyproj.Proj(proj='latlong')

    xs, ys = pyproj.transform(lla, projection, lon.flatten(), lat.flatten())
    xs = xs.reshape(lat.shape)
    ys = ys.reshape(lon.shape)

    # At this point, if all has gone well, xs has every column the same,
    # and ys has every row the same. Maybe they're not exactly the same
    # (due to rounding errors), so we'll average them.
    xs = np.mean(xs, axis=0)
    ys = np.mean(ys, axis=1)

    return reader.import_grids(xs=xs, ys=ys, pressure=plevs,
                               temperature=sim_temp, humidity=sim_humidity,
                               geo_ht=sim_geo_ht,
                               k1=netcdf._k1, k2=netcdf._k2, k3=netcdf._k3,
                               projection=projection)


def sim_weather_plots(sim_weather, lat, lon, los):
    lats = util.gdal_open(lat)
    lons = util.gdal_open(lon)
    llas = np.stack((lats, lons, np.zeros_like(lats)), axis=-1)

    hydro, wet = delay.delay_from_grid(
            sim_weather, llas.reshape(-1, 3), delay.Zenith)
    hydro = hydro.reshape(lats.shape)
    wet = wet.reshape(lats.shape)
    zenith = hydro + wet

    incidence, heading = util.gdal_open(los)
    hydro, wet = delay.delay_from_grid(sim_weather, llas.reshape(-1, 3), incidence.flatten(), raytrace=False)
    hydro = hydro.reshape(lats.shape)
    wet = wet.reshape(lats.shape)
    cmbd_cosine = hydro + wet

    lvs = util.los_to_lv(incidence, heading, lats, lons, np.zeros_like(lats))

    hydro, wet = delay.delay_from_grid(sim_weather, llas.reshape(-1, 3), lvs.reshape(-1, 3), raytrace=True)
    hydro = hydro.reshape(lats.shape)
    wet = wet.reshape(lats.shape)
    cmbd_raytrace = hydro + wet

    return zenith, cmbd_raytrace, cmbd_cosine


def make_plots(zenith, raytrace, cosine):
    plt.imshow(zenith, vmin=1.925, vmax=1.928)
    plt.colorbar()
    plt.savefig('simulation/zenith.pdf', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(4.5,4))

    top = np.nanpercentile(np.stack((zenith, raytrace)), 95)
    bottom = np.nanpercentile(np.stack((zenith, raytrace)), 5)

    fig.add_subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(raytrace, vmin=bottom, vmax=top)
    plt.colorbar()
    plt.title('Raytraced')

    fig.add_subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(cosine, vmin=bottom, vmax=top)
    plt.colorbar()
    plt.title('Projected')

    fig.add_subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(raytrace - cosine, vmin=-0.003, vmax=0.003)
    plt.colorbar()
    plt.title('Difference')

    plt.savefig('simulation/comparison.pdf', bbox_inches='tight')
    plt.show()


def test_statevecs():
    weather = test_weather()
    t = np.load('/home/hogenson/t.npy')
    x, y, z = np.load('/home/hogenson/position.npy')
    vx, vy, vz = np.load('/home/hogenson/velocity.npy')
    lats = util.gdal_open(lat_kriek)
    lons = util.gdal_open(lon_kriek)
    heights = util.gdal_open(height_kriek)
    los = np.stack(util.state_to_los_indiv(t, x, y, z, vx, vy, vz, lats, lons, heights.astype(np.double)), axis=-1)
    # Tiny
    # los *= 15000/np.max(np.linalg.norm(los, axis=-1))
    llas = np.stack((lats, lons, heights), axis=-1)
    return delay.delay_from_grid(weather, llas, los, parallel=True)


def los_ecef_to_lla(los, lats, lons, heights):
    los_ecef = np.moveaxis(los, -1, 0)
    ground_lla = np.stack((lats, lons, heights))
    ground_ecef = np.stack(util.lla2ecef(*ground_lla))
    sensor_ecef = ground_ecef + los_ecef
    sensor_lla = np.stack(util.ecef2lla(*sensor_ecef))
    los_lla = sensor_lla - ground_lla
    return np.moveaxis(los_lla, 0, -1)

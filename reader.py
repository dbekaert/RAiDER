"""Helper utilities for file readers."""


import numpy as np
import scipy
import util


class LinearModel:
    """Generic weather model.

    This model is based upon a linear interpolation scheme for pressure,
    temperature, and relative humidity.
    """
    def __init__(self, lats, lons, heights, pressure, temperature, humidity, k1, k2, k3, scipy_interpolate):
        """Initialize a NetCDFModel."""
        if scipy_interpolate:
            points = np.stack(
                    util.lla2ecef(
                        np.broadcast_to(
                            lats.reshape((1,) + lats.shape),
                            (len(heights),) + lats.shape).flatten(),
                        np.broadcast_to(
                            lons.reshape((1,) + lons.shape),
                            (len(heights),) + lons.shape).flatten(),
                        heights.flatten()), axis=-1)
            nonnan = np.all(np.logical_not(np.isnan(points)), axis=1)
            self._p_inp = scipy.interpolate.LinearNDInterpolator(
                    points[nonnan], np.log(pressure.flatten())[nonnan])
            self._t_inp = scipy.interpolate.LinearNDInterpolator(
                    points[nonnan], temperature.flatten()[nonnan])
            self._h_inp = scipy.interpolate.LinearNDInterpolator(
                    points[nonnan], humidity.flatten()[nonnan])
        else:
            (self._p_inp, self.p_grid, _, _, _), (self._t_inp, self.t_grid, self.new_heights, self.newlons, self.newlats), (self._h_inp, self.h_grid, _, _, _) = _sane_interpolate(
                    lats, lons, heights,
                    (np.log(pressure), temperature, humidity))
            # def interpolate_pressure(pts):
            #     lat, lon, ht = util.ecef2lla(*pts.T)
            #     return self._p_inp_raw(np.stack((ht, lon, lat), axis=-1))
            # def interpolate_temperature(pts):
            #     lat, lon, ht = util.ecef2lla(*pts.T)
            #     return self._t_inp_raw(np.stack((ht, lon, lat), axis=-1))
            # def interpolate_humidity(pts):
            #     lat, lon, ht = util.ecef2lla(*pts.T)
            #     return self._h_inp_raw(np.stack((ht, lon, lat), axis=-1))
            # self._p_inp = interpolate_pressure
            # self._t_inp = interpolate_temperature
            # self._h_inp = interpolate_humidity
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def wet_delay(self, a):
        """Calculate delay at a list of points."""
        temperature = self._t_inp(a)
        humidity = self._h_inp(a)
        e = _find_e(temperature, humidity)

        wet_delay = self.k2*e/temperature + self.k3*e/temperature**2
        return wet_delay

    def hydrostatic_delay(self, a):
        """Calculate hydrostatic delay at a list of points."""
        temperature = self._t_inp(a)
        pressure = np.exp(self._p_inp(a))

        hydro_delay = self.k1*pressure/temperature
        return hydro_delay


def _find_e(temp, rh):
    """Calculate partial pressure of water vapor."""
    # From TRAIN:
    # Could not find the wrf used equation as they appear to be
    # mixed with latent heat etc. Istead I used the equations used
    # in ERA-I (see IFS documentation part 2: Data assimilation
    # (CY25R1)). Calculate saturated water vapour pressure (svp) for
    # water (svpw) using Buck 1881 and for ice (swpi) from Alduchow
    # and Eskridge (1996) euation AERKi

    # TODO: figure out the sources of all these magic numbers and move
    # them somewhere more visible.
    svpw = (6.1121
            * np.exp((17.502*(temp - 273.16))/(240.97 + temp - 273.16)))
    svpi = (6.1121
            * np.exp((22.587*(temp - 273.16))/(273.86 + temp - 273.16)))
    tempbound1 = 273.16 # 0
    tempbound2 = 250.16 # -23

    svp = svpw
    wgt = (temp - tempbound2)/(tempbound1 - tempbound2)
    svp = svpi + (svpw - svpi)*wgt**2
    ix_bound1 = temp > tempbound1
    svp[ix_bound1] = svpw[ix_bound1]
    ix_bound2 = temp < tempbound2
    svp[ix_bound2] = svpi[ix_bound2]

    e = rh/100 * svp * 100

    return e


def _propagate_down(a):
    """Pull real values down to cover NaNs

    a might contain some NaNs which live under real values. We replace
    those NaNs with actual values. a must be a 3D array.
    """
    out = a.copy()
    z, x, y = out.shape
    for i in range(x):
        for j in range(y):
            held = None
            for k in range(z):
                val = out[k][i][j]
                if np.isnan(val) and held is not None:
                    out[k][i][j] = held
                elif not np.isnan(val):
                    held = val
    return out


def _sane_interpolate(lats, lons, heights, values_list):
    num_levels = 2 * heights.shape[0]
    # First, find the maximum height in each column
    max_heights = np.nanmax(heights, axis=0)
    new_top = np.max(max_heights)

    min_heights = np.nanmin(heights, axis=0)
    new_bottom = np.min(min_heights)

    # new_heights = np.linspace(new_bottom, new_top, num_levels)
    new_heights = np.linspace(-50, new_top, num_levels)

    inp_values = [np.zeros((len(new_heights), lats.shape[0], lats.shape[1])) for values in values_list]

    # TODO: do without a for loop
    for iv in range(len(values_list)):
        for x in range(inp_values[iv].shape[1]):
            for y in range(inp_values[iv].shape[2]):
                not_nan = np.logical_not(np.isnan(heights[:,x,y]))
                f = scipy.interpolate.interp1d(heights[:,x,y][not_nan], values_list[iv][:,x,y][not_nan], bounds_error=False)
                inp_values[iv][:,x,y] = f(new_heights)
        inp_values[iv] = _propagate_down(inp_values[iv])

    minlat = np.min(np.min(lats, axis=1))
    maxlat = np.max(np.max(lats, axis=1))

    minlon = np.min(np.min(lons, axis=0))
    maxlon = np.max(np.max(lons, axis=0))

    # TODO: maybe don't hard-code the shape indices?
    # TODO: is 2x too much resolution? (yes)
    newlats = np.linspace(minlat, maxlat, 2*lats.shape[1])
    newlons = np.linspace(minlon, maxlon, 2*lons.shape[0])

    points = np.stack((lons.flatten(), lats.flatten()), axis=-1)

    # Every bug results from traversal order
    # We want indexing to be ij, since we'd like to index as lon, lat
    pts = np.stack(np.meshgrid(newlons, newlats, indexing='ij'), axis=-1).reshape(-1, 2)

    regular_grid = [np.zeros((inp_values[i].shape[0], newlons.size, newlats.size)) for i in range(len(values_list))]

    # TODO: for loops are slow
    for iv in range(len(values_list)):
        for z in range(inp_values[iv].shape[0]):
            f = scipy.interpolate.LinearNDInterpolator(points, inp_values[iv][z].flatten())
            regular_grid[iv][z] = f(pts).reshape(regular_grid[iv][z].shape)

    np.save('regular_grid', regular_grid)

    zs, xs, ys = np.meshgrid(new_heights, newlons, newlats)

    interpolator = [(scipy.interpolate.RegularGridInterpolator((new_heights, newlons, newlats), regular_grid_, bounds_error=False), regular_grid_, new_heights, newlons, newlats) for regular_grid_ in regular_grid]

    #return interpolator

    interps = list()
    for inp, grid, a, b, c in interpolator:
        # Python has some weird behavior here, eh?
        def ggo(interp):
            def go(a):
                lat, lon, ht = util.ecef2lla(*a.T)
                ans = interp(np.stack((ht, lon, lat), axis=-1))
                return ans
            return go
        interps.append((ggo(inp), grid, a, b, c))

    return interps


def import_grids(lats, lons, pressure, temperature, humidity, geo_ht,
                 k1, k2, k3, temp_fill=np.nan, humid_fill=np.nan,
                 geo_ht_fill=np.nan, scipy_interpolate=True):
    """Import weather information to make a weather model object.
    
    This takes in lat, lon, pressure, temperature, humidity in the 3D
    grid format that NetCDF uses, and I imagine might be common
    elsewhere. If other weather models don't make it convenient to use
    this function, we'll need to add some more abstraction. For now,
    this function is only used for NetCDF anyway.
    """
    # In some cases, the pressure goes the wrong way. The way we'd like
    # is from top to bottom, i.e., low pressures to high pressures. If
    # that's not the case, we'll reverse everything.
    if pressure[0] > pressure[1]:
        pressure = pressure[::-1]
        temperature = temperature[::-1]
        humidity = humidity[::-1]
        geo_ht = geo_ht[::-1]
    # Replace the non-useful values by NaN, and fill in values under
    # the topography
    # temps_fixed = _propagate_down(np.where(temperature != temp_fill,
    #                                           temperature, np.nan))
    # humids_fixed = _propagate_down(np.where(humidity != humid_fill,
    #                                            humidity, np.nan))
    # geo_ht_fix = _propagate_down(np.where(geo_ht != geo_ht_fill,
    #                                          geo_ht, np.nan))

    temps_fixed = np.where(temperature != temp_fill, temperature, np.nan)
    humids_fixed = np.where(humidity != humid_fill, humidity, np.nan)
    geo_ht_fix = np.where(geo_ht != geo_ht_fill, geo_ht, np.nan)

    heights = util.geo_to_ht(lats, lons, geo_ht_fix)

    new_plevs = np.repeat(pressure, lats.shape[0] * lats.shape[1]).reshape(-1, lats.shape[0], lats.shape[1])

    return LinearModel(lats=lats, lons=lons, heights=heights, pressure=new_plevs,
                       temperature=temps_fixed, humidity=humids_fixed,
                       k1=k1, k2=k2, k3=k3,
                       scipy_interpolate=scipy_interpolate)

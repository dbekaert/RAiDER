"""Helper utilities for file readers."""


import numpy as np
import pyproj
import scipy
import util


# Minimum z (height) value on the earth
_zmin = -100


def _least_nonzero(a):
    out = np.full(a.shape[1:], np.nan)
    xlim, ylim = out.shape
    zlim = len(a)
    for x in range(xlim):
        for y in range(ylim):
            for z in range(zlim):
                val = a[z][x][y]
                if not np.isnan(val):
                    out[x][y] = val
                    break
    return out


class LinearModel:
    """Generic weather model.

    This model is based upon a linear interpolation scheme for pressure,
    temperature, and relative humidity.
    """
    def __init__(self, xs, ys, heights, pressure, temperature, humidity,
                 k1, k2, k3, projection, scipy_interpolate, humidity_type,
                 zmin):
        """Initialize a NetCDFModel."""
        zmin = -100
        if scipy_interpolate:
            # Add an extra layer below to interpolate below the surface
            if np.min(heights) > zmin:
                new_heights = np.zeros(heights.shape[1:]) + zmin
                new_pressures = _least_nonzero(pressure)
                new_temps = _least_nonzero(temperature)
                new_humids = _least_nonzero(humidity)
                heights = np.concatenate((new_heights[np.newaxis], heights))
                pressure = np.concatenate(
                        (new_pressures[np.newaxis], pressure))
                temperature = np.concatenate(
                        (new_temps[np.newaxis], temperature))
                humidity = np.concatenate((new_humids[np.newaxis], humidity))

            ecef = pyproj.Proj(proj='geocent')

            # Points in native projection
            points_a = np.broadcast_to(xs[np.newaxis,np.newaxis,:],
                                       pressure.shape)
            points_b = np.broadcast_to(ys[np.newaxis,:,np.newaxis],
                                       pressure.shape)
            points_c = heights

            # Points in ecef
            points = np.stack(pyproj.transform(projection, ecef,
                                               points_a.flatten(),
                                               points_b.flatten(),
                                               points_c.flatten()), axis=-1)

            pts_nonnan = np.all(np.logical_not(np.isnan(points)), axis=1)
            both = np.logical_and(np.logical_not(np.isnan(pressure)).flatten(),
                                  pts_nonnan)
            self._p_inp = scipy.interpolate.LinearNDInterpolator(
                    points[both], np.log(pressure.flatten())[both])
            both = np.logical_and(
                    np.logical_not(np.isnan(temperature)).flatten(),
                    pts_nonnan)
            self._t_inp = scipy.interpolate.LinearNDInterpolator(
                    points[both], temperature.flatten()[both])
            both = np.logical_and(np.logical_not(np.isnan(humidity)).flatten(),
                                  pts_nonnan)
            self._h_inp = scipy.interpolate.LinearNDInterpolator(
                    points[both], humidity.flatten()[both])
        else:
            self._p_inp, self._t_inp, self._h_inp = _sane_interpolate(
                    xs, ys, heights, projection,
                    (np.log(pressure), temperature, humidity),
                    zmin=zmin)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.humidity_type = humidity_type

    def wet_delay(self, a):
        """Calculate delay at a list of points."""
        temperature = self._t_inp(a)
        humidity = self._h_inp(a)
        pressure = np.exp(self._p_inp(a))
        # Sometimes we've got it directly
        if self.humidity_type == 'q':
            e = _find_e_from_q(temperature, humidity, pressure)
        elif self.humidity_type == 'rh':
            e = _find_e_from_rh(temperature, humidity)
        else:
            raise ValueError('self.humidity_type should be one of q or rh. It '
                f'was {self.humidityType}')

        wet_delay = self.k2*e/temperature + self.k3*e/temperature**2
        return wet_delay

    def hydrostatic_delay(self, a):
        """Calculate hydrostatic delay at a list of points."""
        temperature = self._t_inp(a)
        pressure = np.exp(self._p_inp(a))

        hydro_delay = self.k1*pressure/temperature
        return hydro_delay


def _find_svp(temp):
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

    return svp * 100


def _find_e_from_q(temp, q, p):
    R_v = 461.524
    R_d = 287.053
    e_s = _find_svp(temp)
    # We have q = w/(w + 1), so w = q/(1 - q)
    w = q/(1 - q)
    e = w*R_v*(p - e_s)/R_d


def _find_e_from_rh(temp, rh):
    """Calculate partial pressure of water vapor."""
    svp = _find_svp(temp)

    e = rh/100 * svp

    return e


def _just_pull_down(a, direction=-1):
    """Pull real values down to cover NaNs

    a might contain some NaNs which live under real values. We replace
    those NaNs with actual values. a must be a 3D array.
    """
    out = a.copy()
    z, x, y = out.shape
    for i in range(x):
        for j in range(y):
            held = None
            if direction == 1:
                r = range(z)
            elif direction == -1:
                r = range(z - 1, -1, -1)
            else:
                raise ValueError(
                        'Unsupported direction. direction should be 1 or -1')
            for k in r:
                val = out[k][i][j]
                if np.isnan(val) and held is not None:
                    out[k][i][j] = held
                elif not np.isnan(val):
                    held = val
    return out


def _propagate_down(a, direction=1):
    out = np.zeros_like(a)
    z, x, y = a.shape
    xs = np.arange(x)
    ys = np.arange(y)
    xgrid, ygrid = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack((xgrid, ygrid), axis=-1)
    for i in range(z):
        nans = np.isnan(a[i])
        nonnan = np.logical_not(nans)
        inpts = points[nonnan]
        apts = a[i][nonnan]
        outpoints = points.reshape(-1, 2)
        try:
            ans = scipy.interpolate.griddata(inpts, apts, outpoints,
                                             method='nearest')
        except ValueError:
            # Likely there aren't any (non-nan) values here, but we'll
            # copy over the whole thing to be safe.
            ans = a[i]
        out[i] = ans.reshape(out[i].shape)
    # I honestly have no idea if this will work
    return _just_pull_down(out)


def _sane_interpolate(xs, ys, heights, projection, values_list, zmin=None):
    if zmin is None:
        zmin = _zmin

    num_levels = 2 * heights.shape[0]
    # First, find the maximum height
    new_top = np.nanmax(heights)

    new_bottom = np.nanmin(heights)

    new_heights = np.linspace(zmin, new_top, num_levels)

    inp_values = [np.zeros((len(new_heights),) + values.shape[1:])
            for values in values_list]

    # TODO: do without a for loop
    for iv in range(len(values_list)):
        for x in range(values_list[iv].shape[1]):
            for y in range(values_list[iv].shape[2]):
                not_nan = np.logical_not(np.isnan(heights[:,x,y]))
                inp_values[iv][:,x,y] = scipy.interpolate.griddata(
                        heights[:,x,y][not_nan],
                        values_list[iv][:,x,y][not_nan],
                        new_heights,
                        method='cubic')
        inp_values[iv] = _propagate_down(inp_values[iv], -1)

    ecef = pyproj.Proj(proj='geocent')

    interps = list()
    for iv in range(len(values_list)):
        # Indexing as height, ys, xs is a bit confusing, but it'll error
        # if the sizes don't match, so we can be sure it's the correct
        # order.
        f = scipy.interpolate.RegularGridInterpolator((new_heights, ys, xs),
                                                      inp_values[iv],
                                                      bounds_error=False)
        # Python has some weird behavior here, eh?
        def ggo(interp):
            def go(pts):
                xs, ys, zs = np.moveaxis(pts, -1, 0)
                a, b, c = pyproj.transform(ecef, projection, xs, ys, zs)
                # Again we index as ys, xs
                llas = np.stack((c, b, a), axis=-1)
                return interp(llas)
            return go
        interps.append(ggo(f))

    return interps


def import_grids(xs, ys, pressure, temperature, humidity, geo_ht,
                 k1, k2, k3, projection, temp_fill=np.nan, humid_fill=np.nan,
                 geo_ht_fill=np.nan, scipy_interpolate=False,
                 humidity_type='rh', zmin=None):
    """Import weather information to make a weather model object.
    
    This takes in lat, lon, pressure, temperature, humidity in the 3D
    grid format that NetCDF uses, and I imagine might be common
    elsewhere. If other weather models don't make it convenient to use
    this function, we'll need to add some more abstraction. For now,
    this function is only used for NetCDF anyway.
    """
    if zmin is None:
        zmin = _zmin

    # In some cases, the pressure goes the wrong way. The way we'd like
    # is from bottom to top, i.e., high pressures to low pressures. If
    # that's not the case, we'll reverse everything.
    if pressure[0] < pressure[1]:
        pressure = pressure[::-1]
        temperature = temperature[::-1]
        humidity = humidity[::-1]
        geo_ht = geo_ht[::-1]

    # Replace the non-useful values by NaN
    temps_fixed = np.where(temperature != temp_fill, temperature, np.nan)
    humids_fixed = np.where(humidity != humid_fill, humidity, np.nan)
    geo_ht_fix = np.where(geo_ht != geo_ht_fill, geo_ht, np.nan)

    # We've got to recover the grid of lat, lon
    xgrid, ygrid = np.meshgrid(xs, ys)
    lla = pyproj.Proj(proj='latlong')
    lons, lats = pyproj.transform(projection, lla, xgrid, ygrid)

    heights = util.geo_to_ht(lats, lons, geo_ht_fix)

    new_plevs = np.broadcast_to(pressure[:,np.newaxis,np.newaxis],
                                heights.shape)

    return LinearModel(xs=xs, ys=ys, heights=heights,
                       pressure=new_plevs,
                       temperature=temps_fixed, humidity=humids_fixed,
                       k1=k1, k2=k2, k3=k3, projection=projection,
                       scipy_interpolate=scipy_interpolate,
                       humidity_type=humidity_type, zmin=zmin)

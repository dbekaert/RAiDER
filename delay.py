"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _step meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity. We could develop a more advanced integrator to deal with these
issues, and probably I will. It goes pretty quickly right now, though.
"""


import multiprocessing
import numpy
import progressbar
import scipy.integrate as integrate
import util


# Step in meters to use when integrating
_step = 15

# Parameters from Hanssen, 2001
_k1 = 0.776 # [K/Pa]
# Should be k2'
_k2 = 0.233 # [K/Pa]
_k3 = 3.75e3 # [K^2/Pa]


class Zenith:
    """Special value indicating a look vector of "zenith"."""
    pass


def _from_generator(g, n):
    """Return a numpy array by taking n elements from g.

    If g returns fewer than n elements, the rest are padded with 0s, but
    you really should just pass in n equal to how many things g will
    return.

    It's like 4 lines and therefore too long to inline everywhere that I
    want to use this function.
    """
    a = numpy.zeros(n)
    for i, el in enumerate(g):
        a[i] = el
    return a


def _common_delay(weather, lat, lon, height, look_vec, rnge):
    """Perform computation common to hydrostatic and dry delay."""
    position = util.lla2ecef(lat, lon, height)
    if look_vec is not Zenith:
        raise NotImplemented
    else:
        corrected_lv = util.lla2ecef(lat, lon, height + 1) - position
    unit_look_vec = corrected_lv / numpy.linalg.norm(corrected_lv)
    t_points = numpy.linspace(0, rnge, rnge / _step)
    def where(t):
        return position + unit_look_vec * t
    return t_points, where


def _work(l):
    """Worker function for integrating delay in a thread."""
    weather, lats, lons, hts, i, j, k = l
    big = 15000
    return (i, j, k,
            hydrostatic_delay(weather, lats[j], lons[k], hts[i],
                              Zenith, big),
            dry_delay(weather, lats[j], lons[k], hts[i], Zenith,
                      big))


def dry_delay(weather, lat, lon, height, look_vec, rnge):
    """Compute dry delay along the look vector."""
    t_points, where = _common_delay(weather, lat, lon, height, look_vec, rnge)

    temp = _from_generator((weather.temperature(*where(t)) for t in t_points),
                           t_points.size)
    rh = _from_generator((weather.rel_humid(*where(t)) for t in t_points),
                         t_points.size)

    svpw = 6.1121*numpy.exp((17.502*(temp - 273.16))/(240.97 + temp - 273.16))
    svpi = 6.1121*numpy.exp((22.587*(temp - 273.16))/(273.86 + temp - 273.16))
    tempbound1 = 273.16 # 0
    tempbound2 = 250.16 # -23
    svp = svpw
    wgt = (temp - tempbound2)/(tempbound1 - tempbound2)
    svp = svpi + (svpw - svpi)*wgt**2
    ix_bound1 = temp > tempbound1
    svp[ix_bound1] = svpw[ix_bound1]
    ix_bound2 = temp < tempbound2
    svp[ix_bound2] = svpi[ix_bound2]
    e = rh/100 * svp

    delay = _k2*e/temp + _k3*e/temp**2

    delay[numpy.isnan(delay)] = 0

    return numpy.trapz(delay, t_points)


def hydrostatic_delay(weather, lat, lon, height, look_vec, rnge):
    """Compute hydrostatic delay along the look vector."""
    t_points, where = _common_delay(weather, lat, lon, height, look_vec, rnge)

    temp = _from_generator((weather.temperature(*where(t)) for t in t_points),
                           t_points.size)
    p = _from_generator((weather.pressure(*where(t)) for t in t_points),
                        t_points.size)

    delay = _k1*p/temp

    delay[numpy.isnan(delay)] = 0

    return numpy.trapz(delay, t_points)


def delay_over_area(weather, lat_min, lat_max, lat_res, lon_min, lon_max,
                    lon_res, ht_min, ht_max, ht_res):
    """Calculate (in parallel!!) the delays over an area."""
    lats = numpy.arange(lat_min, lat_max, lat_res)
    lons = numpy.arange(lon_min, lon_max, lon_res)
    hts = numpy.arange(ht_min, ht_max, ht_res)
    out = numpy.zeros((hts.size, lats.size, lons.size),
                      dtype=[('hydro', 'float64'), ('dry', 'float64')])
    with multiprocessing.Pool() as pool:
        jobs = ((weather, lats, lons, hts, i, j, k)
                for i in range(hts.size)
                for j in range(lats.size)
                for k in range(lons.size))
        num_jobs = hts.size * lats.size * lons.size
        answers = pool.imap_unordered(_work, jobs, chunksize=10)
        bar = progressbar.progressbar(answers,
                                      widgets=[progressbar.Bar(), ' ',
                                               progressbar.ETA()],
                                      max_value=num_jobs)
        for result in bar:
            i, j, k, hydro_delay, dry_delay = result
            out[i][j][k] = (hydro_delay, dry_delay)
    return out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Brett Buzzanga, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import datetime
import shelve

import xml.etree.ElementTree as ET
import numpy as np
from abc import ABC
from scipy.interpolate import interp1d

from RAiDER.utilFcns import (
    cosd, sind, rio_open, enu2ecef, lla2ecef, ecef2enu, ecef2lla
)
from RAiDER.constants import _ZREF


class LOS(ABC):
    '''
    LOS Class definition for handling look vectors
    '''
    def __init__(self):
        self._lats, self._lons, self._heights = None, None, None
        self._look_vecs = None
        self._ray_trace = False
        self._is_zenith = False
        self._is_projected = False


    def setPoints(self, lats, lons=None, heights=None):
        '''Set the pixel locations'''
        if (lats is None) and (self._lats is None):
            raise RuntimeError("You haven't given any point locations yet")

        # Will overwrite points by default
        if lons is None:
            llh = lats  # assume points are [lats lons heights]
            self._lats = llh[..., 0]
            self._lons = llh[..., 1]
            self._heights = llh[..., 2]
        elif heights is None:
            self._lats = lats
            self._lons = lons
            self._heights = np.zeros((len(lats), 1))
        else:
            self._lats = lats
            self._lons = lons
            self._heights = heights


    def setTime(self, dt):
        self._time = dt


    def is_Zenith(self):
        return self._is_zenith


    def is_Projected(self):
        return self._is_projected


    def ray_trace(self):
        return self._ray_trace


class Zenith(LOS):
    """
    Class definition for a "Zenith" object.
    """
    def __init__(self):
        super().__init__()
        self._is_zenith = True


    def setLookVectors(self):
        '''Set point locations and calculate Zenith look vectors'''
        if self._lats is None:
            raise ValueError('Target points not set')
        if self._look_vecs is None:
            self._look_vecs = getZenithLookVecs(self._lats, self._lons, self._heights)


    def __call__(self, delays):
        '''Placeholder method for consistency with the other classes'''
        return delays


class Conventional(LOS):
    """
    Special value indicating that the zenith delay will
    be projected using the standard cos(inc) scaling.
    """
    def __init__(self, filename=None, los_convention='isce', time=None, pad=600):
        super().__init__()
        self._file = filename
        self._time = time
        self._pad  = pad
        self._is_projected = True
        self._convention   = los_convention
        if self._convention.lower() != 'isce':
            raise NotImplementedError()


    def __call__(self, delays):
        '''
        Read the LOS file and convert it to look vectors
        '''
        if self._lats is None:
            raise ValueError('Target points not set')
        if self._file is None:
            raise ValueError('LOS file not set')

        try:
            # if an ISCE-style los file is passed open it with GDAL
            LOS_enu = inc_hd_to_enu(*rio_open(self._file))

        except (OSError, TypeError):
            # Otherwise, treat it as an orbit / statevector file
            svs = np.stack(
                get_sv(self._file, self._time, self._pad), axis=-1
            )
            LOS_enu = state_to_los(svs,
                                   [self._lats, self._lons, self._heights],
                                   )

        if delays.shape == LOS_enu.shape:
            return delays / LOS_enu
        else:
            return delays / LOS_enu[..., -1]


class Raytracing(LOS):
    """
    Special value indicating that full raytracing will be
    used to calculate slant delays.

    Get unit look vectors pointing from the ground (target) pixels to the sensor,
    or to Zenith. Can be accomplished using an ISCE-style 2-band LOS file or a
    file containing orbital statevectors.

    *NOTE*:
    These line-of-sight vectors will NOT match ordinary LOS vectors for InSAR
    because they are in an ECEF reference frame instead of a local ENU. This is done
    because the construction of rays is done in ECEF rather than the local ENU.

    Args:
    ----------
    time: python datetime                  - user-requested query time. Must be
                                             compatible with the orbit file passed.
                                             Only required for a statevector file.
    pad: int                               - integer number of seconds to pad around
                                             the user-specified time; default 10 min
                                             Only required for a statevector file.

    Returns:
    -------
    ndarray - an <in_shape> x 3 array of unit look vectors, defined in
            an Earth-centered, earth-fixed reference frame (ECEF).
            Convention is vectors point from the target pixel to the
            sensor.
    ndarray - array of <in_shape> of the distnce from the surface to
            the top of the troposphere (denoted by zref)

    Example:
    --------
    >>> from RAiDER.losreader import Raytracing
    >>> import numpy as np
    """
    def __init__(self, filename=None, los_convention='isce', time=None, look_dir = 'right', pad=600):
        '''read in and parse a statevector file'''
        super().__init__()
        self._ray_trace = True
        self._file = filename
        self._time = time
        self._pad  = pad
        self._convention = los_convention
        self._orbit = None
        if self._convention.lower() != 'isce':
            raise NotImplementedError()

        # ISCE3 data structures
        import isce3.ext.isce3 as isce
        if self._time is not None:
            # __call__ called in checkArgs; keep for modularity
            self._orbit = get_orbit(self._file, self._time, pad=pad)
        self._elp = isce.core.Ellipsoid()
        self._dop = isce.core.LUT2d()
        if look_dir.lower() == "right":
            self._look_dir = isce.core.LookSide.Right
        elif look_dir.lower() == "left":
            self._look_dir = isce.core.LookSide.Left
        else:
            raise RuntimeError(f"Unknown look direction: {look_dir}")
    

    def getSensorDirection(self):
        if self._orbit is None:
            raise ValueError('The orbit has not been set')
        z = self._orbit.position[:,2]
        t = self._orbit.time
        start = np.argmin(t)
        end = np.argmax(t)
        if z[start] > z[end]:
            return 'desc'
        else:
            return 'asc'
    

    def getLookDirection(self):
        return self._look_dir

    # Called in checkArgs
    def setTime(self, time, pad=600):
        self._time = time
        self._orbit = get_orbit(self._file, self._time, pad=pad)


    def getLookVectors(self, ht, llh, xyz, yy):
        '''
        Calculate look vectors for raytracing
        '''
        # TODO - Modify when isce3 vectorization is available
        los = np.full(yy.shape + (3,), np.nan)
        llh[0] = np.deg2rad(llh[0])
        llh[1] = np.deg2rad(llh[1])

        import isce3.ext.isce3 as isce

        for ii in range(yy.shape[0]):
            for jj in range(yy.shape[1]):
                inp = np.array([llh[0][ii, jj], llh[1][ii, jj], ht])
                inp_xyz = xyz[ii, jj, :]

                if any(np.isnan(inp)) or any(np.isnan(inp_xyz)):
                    continue

                # Wavelength does not matter for
                try:
                    aztime, slant_range = isce.geometry.geo2rdr(
                        inp, self._elp, self._orbit, self._dop, 0.06, self._look_dir,
                        threshold=1.0e-7,
                        maxiter=30,
                        delta_range=10.0)
                    sat_xyz, _ = self._orbit.interpolate(aztime)
                    los[ii, jj, :] = (sat_xyz - inp_xyz) / slant_range
                except Exception as e:
                    los[ii, jj, :] = np.nan
        return los


    def getIntersectionWithHeight(self, height):
        """
        This function computes the intersection point of a ray at a height
        level
        """
        # We just leverage the same code as finding top of atmosphere here
        return getTopOfAtmosphere(self._xyz, self._look_vecs, height)


    def getIntersectionWithLevels(self, levels):
        """
        This function returns the points at which rays intersect the
        given height levels. This way we have same number of points in
        each ray and only at level transitions.

        For targets that are above a given height level, the ray points are set
        to nan to indicate that it does not contribute to the integration of
        rays.

        Output:
            rays: (self._lats.shape, len(levels),  3)
        """
        rays = np.zeros(list(self._lats.shape) + [len(levels), 3])

        # This can be further vectorized, if there is enough memory
        for ind, z in enumerate(levels):
            rays[..., ind, :] = self.getIntersectionWithHeight(z)

            # Set pixels above level to nan
            value = rays[..., ind, :]
            value[self._heights > z, :] = np.nan

        return rays


    def calculateDelays(self, delays):
        '''
        Here "delays" is point-wise delays (i.e. refractivities), not
        integrated ZTD/STD.
        '''
        # Create rays  (Use getIntersectionWithLevels above)
        # Interpolate delays to rays
        # Integrate along rays
        # Return STD
        raise NotImplementedError


def getZenithLookVecs(lats, lons, heights):
    '''
    Returns look vectors when Zenith is used.

    Args:
        lats/lons/heights (ndarray):  - Numpy arrays containing WGS-84 target locations

    Returns:
        zenLookVecs (ndarray):        - (in_shape) x 3 unit look vectors in an ECEF reference frame
    '''
    x = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    y = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    z = np.sin(np.radians(lats))

    return np.stack([x, y, z], axis=-1)


def get_sv(los_file, ref_time, pad):
    """
    Read an LOS file and return orbital state vectors

    Args:
        los_file (str):     - user-passed file containing either look
                              vectors or statevectors for the sensor
        ref_time (datetime):- User-requested datetime; if not encompassed
                              by the orbit times will raise a ValueError
        pad (int):          - number of seconds to keep around the
                              requested time (should be about 600 seconds)

    Returns:
        svs (list of ndarrays): - the times, x/y/z positions and velocities
        of the sensor for the given window around the reference time
    """
    try:
        svs = read_txt_file(los_file)
    except (ValueError, TypeError):
        try:
            if isinstance(los_file, list):
                los_file = pick_ESA_orbit_file(los_file, ref_time)

            svs = read_ESA_Orbit_file(los_file)

        except BaseException:
            try:
                svs = read_shelve(los_file)
            except BaseException:
                raise ValueError(
                    'get_sv: I cannot parse the statevector file {}'.format(los_file)
                )

    if ref_time:
        idx = cut_times(svs[0], ref_time, pad=pad)
        svs = [d[idx] for d in svs]

    return svs


def inc_hd_to_enu(incidence, heading):
    '''
    Convert incidence and heading to line-of-sight vectors from the ground to the top of
    the troposphere.

    Args:
        incidence: ndarray	       - incidence angle in deg from vertical
        heading: ndarray 	       - heading angle in deg clockwise from north
        lats/lons/heights: ndarray - WGS84 ellipsoidal target (ground pixel) locations

    Returns:
        LOS: ndarray  - (input_shape) x 3 array of unit look vectors in local ENU

    Algorithm referenced from http://earthdef.caltech.edu/boards/4/topics/327
    '''
    if np.any(incidence < 0):
        raise ValueError('inc_hd_to_enu: Incidence angle cannot be less than 0')

    east = sind(incidence) * cosd(heading + 90)
    north = sind(incidence) * sind(heading + 90)
    up = cosd(incidence)

    return np.stack((east, north, up), axis=-1)


def read_shelve(filename):
    # TODO: docstring and unit tests
    with shelve.open(filename, 'r') as db:
        obj = db['frame']

    numSV = len(obj.orbit.stateVectors)
    if numSV == 0:
        raise ValueError('read_shelve: the file has not statevectors')

    t = []
    x = np.ones(numSV)
    y = np.ones(numSV)
    z = np.ones(numSV)
    vx = np.ones(numSV)
    vy = np.ones(numSV)
    vz = np.ones(numSV)

    for i, st in enumerate(obj.orbit.stateVectors):
        t.append(st.time)
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]

    t = np.array(t)
    return t, x, y, z, vx, vy, vz


def read_txt_file(filename):
    '''
    Read a 7-column text file containing orbit statevectors. Time
    should be denoted as integer time in seconds since the reference
    epoch (user-requested time).

    Args:
        filename (str): - user-supplied space-delimited text file with no header
                        containing orbital statevectors as 7 columns:
                        - time in seconds since the user-supplied epoch
                        - x / y / z locations in ECEF cartesian coordinates
                        - vx / vy / vz velocities in m/s in ECEF coordinates
    Returns:
        svs (list):     - a length-7 list of numpy vectors containing the above
                        variables
    '''
    t = list()
    x = list()
    y = list()
    z = list()
    vx = list()
    vy = list()
    vz = list()
    with open(filename, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split()
                t_ = datetime.datetime.fromisoformat(parts[0])
                x_, y_, z_, vx_, vy_, vz_ = [float(t) for t in parts[1:]]
            except ValueError:
                raise ValueError(
                    "I need {} to be a 7 column text file, with ".format(filename) +
                    "columns t, x, y, z, vx, vy, vz (Couldn't parse line " +
                    "{})".format(repr(line)))
            t.append(t_)
            x.append(x_)
            y.append(y_)
            z.append(z_)
            vx.append(vx_)
            vy.append(vy_)
            vz.append(vz_)

    if len(t) < 4:
        raise ValueError('read_txt_file: File {} does not have enough statevectors'.format(filename))

    return [np.array(a) for a in [t, x, y, z, vx, vy, vz]]


def read_ESA_Orbit_file(filename):
    '''
    Read orbit data from an orbit file supplied by ESA

    Args:
    ----------
    filename: str             - string of the orbit filename

    Returns:
    -------
    t: Nt x 1 ndarray   - a numpy vector with Nt elements containing time
                          in python datetime
    x, y, z: Nt x 1 ndarrays    - x/y/z positions of the sensor at the times t
    vx, vy, vz: Nt x 1 ndarrays - x/y/z velocities of the sensor at the times t
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    data_block = root[1]
    numOSV = len(data_block[0])

    t = []
    x = np.ones(numOSV)
    y = np.ones(numOSV)
    z = np.ones(numOSV)
    vx = np.ones(numOSV)
    vy = np.ones(numOSV)
    vz = np.ones(numOSV)

    for i, st in enumerate(data_block[0]):
        t.append(
            datetime.datetime.strptime(
                st[1].text,
                'UTC=%Y-%m-%dT%H:%M:%S.%f'
            )
        )

        x[i] = float(st[4].text)
        y[i] = float(st[5].text)
        z[i] = float(st[6].text)
        vx[i] = float(st[7].text)
        vy[i] = float(st[8].text)
        vz[i] = float(st[9].text)
    t = np.array(t)
    return [t, x, y, z, vx, vy, vz]


def pick_ESA_orbit_file(list_files:list, ref_time:datetime.datetime):
    """ From list of .EOF orbit files, pick the one that contains 'ref_time' """
    orb_file = None
    for path in list_files:
        f  = os.path.basename(path)
        t0 = datetime.datetime.strptime(f.split('_')[6].lstrip('V'), '%Y%m%dT%H%M%S')
        t1 = datetime.datetime.strptime(f.split('_')[7].rstrip('.EOF'), '%Y%m%dT%H%M%S')
        if t0 < ref_time < t1:
            orb_file = path
            break

    assert orb_file, 'Given orbit files did not match given date/time'

    return path


############################
def state_to_los(svs, llh_targets):
    '''
    Converts information from a state vector for a satellite orbit, given in terms of
    position and velocity, to line-of-sight information at each (lon,lat, height)
    coordinate requested by the user.
    Args:
    ----------
    svs            - t, x, y, z, vx, vy, vz - time, position, and velocity in ECEF of the sensor
    llh_targets    - lats, lons, heights - Ellipsoidal (WGS84) positions of target ground pixels

    Returns:
    -------
    LOS 			- * x 3 matrix of LOS unit vectors in ECEF (*not* ENU)
    Example:
    >>> import datetime
    >>> import numpy
    >>> from RAiDER.utilFcns import rio_open
    >>> import RAiDER.losreader as losr
    >>> lats, lons, heights = np.array([-76.1]), np.array([36.83]), np.array([0])
    >>> time = datetime.datetime(2018,11,12,23,0,0)
    >>> # download the orbit file beforehand
    >>> esa_orbit_file = 'S1A_OPER_AUX_POEORB_OPOD_20181203T120749_V20181112T225942_20181114T005942.EOF'
    >>> svs = losr.read_ESA_Orbit_file(esa_orbit_file)
    >>> LOS = losr.state_to_los(*svs, [lats, lons, heights], xyz)
    '''
    # check the inputs
    if np.min(svs.shape) < 4:
        raise RuntimeError(
            'state_to_los: At least 4 state vectors are required'
            ' for orbit interpolation'
        )

    # Convert svs to isce3 orbit
    import isce3.ext.isce3 as isce
    orb = isce.core.Orbit([
        isce.core.StateVector(
            isce.core.DateTime(row[0]),
            row[1:4], row[4:7]
        ) for row in svs
    ])

    # Flatten the input array for convenience
    in_shape   = llh_targets[0].shape
    target_llh = np.stack([x.flatten() for x in llh_targets], axis=-1)
    Npts       = len(target_llh)

    # Iterate through targets and compute LOS
    los_ang, _ = get_radar_pos(target_llh, orb)
    los_factor = np.cos(np.deg2rad(los_ang)).reshape(in_shape)
    return los_factor


def cut_times(times, ref_time, pad):
    """
    Slice the orbit file around the reference aquisition time. This is done
    by default using a three-hour window, which for Sentinel-1 empirically
    works out to be roughly the largest window allowed by the orbit time.
    Args:
    ----------
    times: Nt x 1 ndarray     - Vector of orbit times as datetime
    ref_time: datetime        - Reference time
    pad: int                  - integer time in seconds to use as padding
    Returns:
    -------
    idx: Nt x 1 logical ndarray - a mask of times within the padded request time.
    """
    diff = np.array(
        [(x - ref_time).total_seconds() for x in times]
    )
    return np.abs(diff) < pad


def get_radar_pos(llh, orb):
    '''
    Calculate the coordinate of the sensor in ECEF at the time corresponding to ***.

    Args:
    ----------
    orb: isce3.core.Orbit   - Nt x 7 matrix of statevectors: [t x y z vx vy vz]
    llh: ndarray   - position of the target in LLH
    out: str    - either lookangle or ecef for vector

    Returns:
    -------
    los: ndarray  - Satellite incidence angle
    sr:  ndarray  - Slant range in meters
    '''

    num_iteration = 30
    residual_threshold = 1.0e-7

    # Get xyz positions of targets here from lat/lon/height
    targ_xyz = np.stack(
        lla2ecef(llh[:, 0], llh[:, 1], llh[:, 2]), axis=-1
    )

    # Get some isce3 constants for this inversion
    import isce3.ext.isce3 as isce
    # TODO - Assuming right-looking for now
    elp = isce.core.Ellipsoid()
    dop = isce.core.LUT2d()
    look = isce.core.LookSide.Right

    # Iterate for each point
    # TODO - vectorize / parallelize
    sr = np.empty((llh.shape[0],), dtype=np.float64)
    output = np.empty((llh.shape[0],), dtype=np.float64)

    for ind, pt in enumerate(llh):
        if not any(np.isnan(pt)):
            # ISCE3 always uses xy convention
            inp = np.array([np.deg2rad(pt[1]),
                            np.deg2rad(pt[0]),
                            pt[2]])
            # Local normal vector
            nv = elp.n_vector(inp[0], inp[1])

            # Wavelength does not matter  for zero doppler
            try:
                aztime, slant_range = isce.geometry.geo2rdr(
                    inp, elp, orb, dop, 0.06, look,
                    threshold=residual_threshold,
                    maxiter=num_iteration,
                    delta_range=10.0)
                sat_xyz, _ = orb.interpolate(aztime)
                sr[ind] = slant_range


                delta = sat_xyz - targ_xyz[ind, :]

                # TODO - if we only ever need cos(lookang),
                # skip the arccos here and cos above
                delta = delta / np.linalg.norm(delta)
                output[ind] = np.rad2deg(
                    np.arccos(np.dot(delta, nv))
                )

            except Exception as e:
                raise e

        # in case nans in hgt field
        else:
            sr[ind] = np.nan
            output[ind, ...] = np.nan

    return output, sr


def getTopOfAtmosphere(xyz, look_vecs, toaheight, factor=None):
    """
    Get ray intersection at given height.

    We use simple Newton-Raphson for this computation. This cannot be done
    exactly since closed form expression from xyz to llh is super compliated.

    If a factor (cos of inc angle) is provided - iterations are lot faster.
    If factor is not provided solutions converges to
        - 0.01 mm at heights near zero in 10 iterations
        - 10 cm at heights above 40km in 10 iterations

    If factor is know, we converge in 3 iterations to less than a micron.
    """
    if factor is not None:
        maxIter = 3
    else:
        maxIter = 10
        factor = 1.

    # Guess top point
    pos = xyz + toaheight * look_vecs

    for _ in range(maxIter):
        pos_llh = ecef2lla(pos[..., 0], pos[..., 1], pos[..., 2])
        pos = pos + look_vecs * ((toaheight - pos_llh[2])/factor)[..., None]

    return pos


def get_orbit(orbit_file, ref_time, pad):
    '''
    Returns state vectors from an orbit file
    orbit file (str):   - user-passed file containing statevectors
                          for the sensor (can download with sentineleof libray)
    pad (int):          - number of seconds to keep around the
                          requested time (should be about 600 seconds)

    '''
    # First load the state vectors into an isce orbit
    import isce3.ext.isce3 as isce

    svs   = np.stack(get_sv(orbit_file, ref_time, pad), axis=-1)
    svs_i = []
    # format for ISCE
    for sv in svs:
       sv = isce.core.StateVector(isce.core.DateTime(sv[0]), sv[1:4], sv[4:7])
       svs_i.append(sv)

    orb = isce.core.Orbit(svs_i)

    return orb
import numpy as np

import os.path
import shelve
import utils.util as util


def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights):
    import Geo2rdr

    real_shape = lats.shape
    lats = lats.flatten()
    lons = lons.flatten()
    heights = heights.flatten()

    geo2rdr_obj = Geo2rdr.PyGeo2rdr()
    geo2rdr_obj.set_orbit(t, x, y, z, vx, vy, vz)

    loss = np.zeros((3, len(lats)))
    slant_ranges = np.zeros_like(lats)

    for i, (lat, lon, height) in enumerate(zip(lats, lons, heights)):
        height_array = np.array(((height,),))

        # Geo2rdr is picky about the type of height
        height_array = height_array.astype(np.double)

        geo2rdr_obj.set_geo_coordinate(np.radians(lon),
                                       np.radians(lat),
                                       1, 1,
                                       height_array)
        # compute the radar coordinate for each geo coordinate
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        los_x, los_y, los_z = geo2rdr_obj.get_los()
        loss[:, i] = los_x, los_y, los_z

        # get back the slant ranges
        slant_range = geo2rdr_obj.get_slant_range()
        slant_ranges[i] = slant_range

    los = loss * slant_ranges

    # Have to think about traversal order here. It's easy, though, since
    # in both orders xs come first, followed by all ys, followed by all
    # zs.
    return los.reshape((3,) + real_shape)


def read_shelve(filename):
    import isce
    import iscesys.Component.ProductManager

    with shelve.open(filename, 'r') as db:
        obj = db['frame']

    numSV = len(obj.orbit.stateVectors)

    t = np.ones(numSV)
    x = np.ones(numSV)
    y = np.ones(numSV)
    z = np.ones(numSV)
    vx = np.ones(numSV)
    vy = np.ones(numSV)
    vz = np.ones(numSV)

    for i, st in enumerate(obj.orbit.stateVectors):
        t[i] = st.time.second + st.time.minute*60.0
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]

    return t, x, y, z, vx, vy, vz


def read_txt_file(filename):
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
                t_, x_, y_, z_, vx_, vy_, vz_ = line.split()
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
    return (np.array(t), np.array(x), np.array(y), np.array(z), np.array(vx),
            np.array(vy), np.array(vz))


def read_xml_file(filename):
    import isce
    import iscesys.Component.ProductManager

    pm = iscesys.Component.ProductManager.ProductManager()
    pm.configure()

    obj = pm.loadProduct(filename)

    numSV = len(obj.orbit.stateVectors)

    t = np.ones(numSV)
    x = np.ones(numSV)
    y = np.ones(numSV)
    z = np.ones(numSV)
    vx = np.ones(numSV)
    vy = np.ones(numSV)
    vz = np.ones(numSV)

    for i, st in enumerate(obj.orbit.stateVectors):
        t[i] = st.time.second + st.time.minute*60.0
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]

    return t, x, y, z, vx, vy, vz


def infer_sv(los_file, lats, lons, heights):
    """Infer the type of file to read, then read an LOS file."""
    _, ext = os.path.splitext(los_file)
    if ext == '.txt':
        svs = read_txt_file(los_file)
    elif ext == '.xml':
        svs = read_xml_file(los_file)
    else:
        # Here's where things get complicated... Either it's a shelve
        # file or the user messed up. For now we'll just try to read it
        # as a shelve file, and throw whatever error that does, although
        # the message might be sometimes misleading.
        svs = read_shelve(los_file)
    LOSs = state_to_los(*svs, lats = lats, lons = lons, heights = heights)
    return LOSs

def los_to_lv(incidence, heading, lats, lons, heights, zref, ranges=None):
    # I'm looking at http://earthdef.caltech.edu/boards/4/topics/327
    a_0 = incidence
    a_1 = heading

    east = util.sind(a_0)*util.cosd(a_1 + 90)
    north = util.sind(a_0)*util.sind(a_1 + 90)
    up = util.cosd(a_0)

    east, north, up = np.stack((east, north, up))

    # Pick reasonable range to top of troposphere if not provided
    if ranges is None:
        ranges = (zref - heights) / up

    # Scale look vectors by range
    east, north, up = np.stack((east, north, up)) * ranges

    x, y, z = util.enu2ecef(
            east.flatten(), north.flatten(), up.flatten(), lats.flatten(),
            lons.flatten(), heights.flatten())

    los = (np.stack((x, y, z), axis=-1)
           - np.stack(util.lla2ecef(
               lats.flatten(), lons.flatten(), heights.flatten()), axis=-1))
    los = los.reshape(east.shape + (3,))

    return los


def infer_los(los, lats, lons, heights, zref):
    '''
    Helper function to deal with various LOS files supplied
    '''

    los_type, los_file = los

    if los_type == 'sv':
        LOS = infer_sv(los_file, lats, lons, heights)

    if los_type == 'los':
        incidence, heading = util.gdal_open(los_file)
        LOS = los_to_lv(incidence, heading, lats, lons, heights, zref)

    return LOS

    raise ValueError("Unsupported los type '{}'".format(los_type))

import os.path
import shelve
try:
    import Geo2rdr
except ImportError as e:
    Geo2rdr_error = e
try:
    import isce
except ImportError as e:
    isce_error = e
else:
    import iscesys.Component.ProductManager


def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights):
    try:
        Geo2rdr
    except NameError:
        raise Geo2rdr_error

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
    try:
        isce
    except NameError:
        raise isce_error
    
    with shelve.open(filename) as db:
        obj=db['frame']
    
    numSV = len(obj.orbit.stateVectors)
    
    t = np.ones(numSV)
    x = np.ones(numSV)
    y = np.ones(numSV)
    z = np.ones(numSV)
    vx = np.ones(numSV)
    vy = np.ones(numSV)
    vz = np.ones(numSV)
    
    for i,st in enumerate(obj.orbit.stateVectors):
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
                raise ValueError(f"I need {filename} to be a 7 column text file, with columns t, x, y, z, vx, vy, vz (Couldn't parse line {repr(line)})")
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
    try:
        isce
    except NameError:
        raise isce_error

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

    for i,st in enumerate(obj.orbit.stateVectors):
        #tt = st.time
        #t[i] = datetime2year(tt)
        t[i] = st.time.second + st.time.minute*60.0
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]


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
    return state_to_los(*svs, lats, lons, heights)

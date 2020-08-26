import argparse

import numpy as np
import pyproj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from RAiDER.losreader import read_ESA_Orbit_file
from RAiDER.rays import OrbitLVGenerator, ZenithLVGenerator
from RAiDER.utilFcns import gdal_open


def lla2ecef(lat, lon, alt):
    t = pyproj.Transformer.from_proj(
        pyproj.Proj(proj='latlong'),
        pyproj.Proj(proj='geocent'),
        always_xy=True
    )
    return t.transform(lon, lat, alt)


def plot_earth(ax, surface=False, detail=10):
    # Using ECEF coordinates

    u = np.linspace(0, 2 * np.pi, detail)
    v = np.linspace(0, np.pi, detail)
    x = 6378137.0 * np.outer(np.cos(u), np.sin(v))
    y = 6378137.0 * np.outer(np.sin(u), np.sin(v))
    z = 6356752.314245179 * np.outer(np.ones(np.size(u)), np.cos(v))

    if surface:
        ax.plot_surface(x, y, z, color='b', zorder=1)
    else:
        ax.plot_wireframe(x, y, z, color='b', zorder=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument("--orbit", help="Generate look vectors using an orbit file")
    p.add_argument("--points", help="Use query points from lat/lon files", nargs=2, metavar=("latfile", "lonfile"))
    p.add_argument("--force-plot", help="Try to plot the points even though performance will be degraded", action="store_true", dest="force")
    p.add_argument("--earth-detail", help="Resolution of the earth wireframe", type=int, dest="detail")
    p.add_argument("--earth-surface", help="Render the earth as a surface instead of a wireframe", action="store_true", dest="surface")

    args = p.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if args.orbit:
        states = read_ESA_Orbit_file(args.orbit)
        gen = OrbitLVGenerator(states)

        orbit_x, orbit_y, orbit_z = states.x[:50], states.y[:50], states.z[:50]
        ax.scatter3D(orbit_x, orbit_y, orbit_z, c="r")
    else:
        gen = ZenithLVGenerator()

    if args.points:
        lats, _, _ = gdal_open(args.points[0], returnProj=True)
        lons, _, _ = gdal_open(args.points[1], returnProj=True)

        lats = lats.flatten()
        lons = lons.flatten()

        if lats.shape[0] > 100 and not args.force:
            raise Exception(
                "Too many points provided! The plot will likely not function "
                "very well, use '--force-plot' to override this warning and "
                "plot anyway."
            )

        llh = np.stack((lats, lons, np.zeros(lats.shape)), axis=-1)
    else:
        llh = np.array([
            [40, -80., 0.],
            [40, -85., 0.],
            [40, -75., 0.],
            [45, -85., 0.],
            [45, -80., 0.],
            [45, -75., 0.],
            [35, -85., 0.],
            [35, -80., 0.],
            [35, -75., 0.]
        ])

    look = gen.generate(llh)

    ground_x, ground_y, ground_z = lla2ecef(llh[..., 0], llh[..., 1], llh[..., 2])
    look_x, look_y, look_z = look[..., 0], look[..., 1], look[..., 2]

    plot_earth(ax, args.surface, args.detail)
    ax.scatter3D(ground_x, ground_y, ground_z, c="g")
    for i in range(llh.shape[0]):
        ax.plot3D(
            np.array([ground_x[i], ground_x[i] + look_x[i] * 2_000_000]),
            np.array([ground_y[i], ground_y[i] + look_y[i] * 2_000_000]),
            np.array([ground_z[i], ground_z[i] + look_z[i] * 2_000_000]),
            color="gray"
        )

    plt.show()

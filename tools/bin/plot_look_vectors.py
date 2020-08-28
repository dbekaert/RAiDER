import argparse

import numpy as np
import pyproj

from RAiDER.losreader import read_ESA_Orbit_file
from RAiDER.makePoints import intersect_altitude, makePoints1D
from RAiDER.rays import OrbitLVGenerator, ZenithLVGenerator
from RAiDER.utilFcns import gdal_open

try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = None


def lla2ecef(lat, lon, alt):
    t = pyproj.Transformer.from_proj(
        pyproj.Proj(proj='latlong'),
        pyproj.Proj(proj='geocent'),
        always_xy=True
    )
    return t.transform(lon, lat, alt)


class MPLPlotter():
    def __init__(self):
        assert plt is not None, "Matplotlib not installed!"

        self.fig = plt.figure()
        self.ax = self.fig.add_axes((0, 0, 1, 1), projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.colormap = {

        }

    def show(self):
        plt.show()

    def add_scatter(self, x, y, z, color):
        self.ax.scatter3D(x, y, z, c=color)

    def add_line(self, x, y, z, color):
        self.ax.plot(x, y, z, c=color)

    def add_surface(self, x, y, z, color):
        self.ax.plot_surface(x, y, z, color=color)

    def add_wireframe(self, x, y, z, color):
        self.ax.plot_wireframe(x, y, z, color=color)


def generate_earth(surface=False, detail=10):
    # Using ECEF coordinates

    semimajor = 6378137.0
    semiminor = 6356752.3142
    u = np.linspace(0, 2 * np.pi, detail)
    v = np.linspace(0, np.pi, detail)
    x = semimajor * np.outer(np.cos(u), np.sin(v))
    y = semimajor * np.outer(np.sin(u), np.sin(v))
    z = semiminor * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument("--orbit", help="Generate look vectors using an orbit file")
    p.add_argument("--points", help="Use query points from lat/lon files", nargs=2, metavar=("latfile", "lonfile"))
    p.add_argument("--force-plot", help="Try to plot the points even though performance will be degraded", action="store_true", dest="force")
    p.add_argument("--earth-detail", help="Resolution of the earth wireframe", type=int, dest="detail")
    p.add_argument("--earth-surface", help="Render the earth as a surface instead of a wireframe", action="store_true", dest="surface")

    args = p.parse_args()

    # Could add other plotters as well. I tried plotly, but I couldn't get it
    # to cooperate on hardly anything, so I gave up.
    plotter = MPLPlotter()

    if args.orbit:
        states = read_ESA_Orbit_file(args.orbit)
        gen = OrbitLVGenerator(states)

        idx = slice(None, 50, 1)
        orbit_x, orbit_y, orbit_z = states.x[idx], states.y[idx], states.z[idx]
        plotter.add_scatter(orbit_x, orbit_y, orbit_z, color="red")
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
            [35, -75., 0.],
        ])
        llh = np.stack((
            np.arange(-90, 90, 10),
            np.arange(-180, 180, 20),
            np.zeros(18)
        ), axis=-1)

    look = gen.generate(llh)

    ground_x, ground_y, ground_z = lla2ecef(llh[..., 0], llh[..., 1], llh[..., 2])
    ground = np.stack((ground_x, ground_y, ground_z), axis=-1)
    look_x, look_y, look_z = look[..., 0], look[..., 1], look[..., 2]

    tropo = intersect_altitude(ground, look, 3_000_000.)

    rays = makePoints1D(3_000_000., ground, look, 100_000.)

    plotter.add_wireframe(*generate_earth(args.surface, args.detail), color="blue")
    plotter.add_scatter(ground_x, ground_y, ground_z, color="green")
    plotter.add_scatter(tropo[..., 0], tropo[..., 1], tropo[..., 2], color="orange")

    for ray in rays:
        plotter.add_line(ray[0], ray[1], ray[2], color="green")

    plotter.show()

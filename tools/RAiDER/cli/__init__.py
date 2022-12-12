import os
from RAiDER.constants import _ZREF, _CUBE_SPACING_IN_M
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DEFAULT_DICT = dict(
        look_dir='right',
        date_start=None,
        date_end=None,
        date_step=None,
        date_list=None,
        time=None,
        end_time=None,
        weather_model=None,
        lat_file=None,
        lon_file=None,
        station_file=None,
        bounding_box=None,
        geocoded_file=None,
        dem=None,
        use_dem_latlon=False,
        height_levels=None,
        height_file_rdr=None,
        ray_trace=False,
        zref=_ZREF,
        cube_spacing_in_m=_CUBE_SPACING_IN_M,  # TODO - Where are these parsed?
        los_file=None,
        los_convention='isce',
        los_cube=None,
        orbit_file=None,
        verbose=True,
        raster_format='GTiff',
        output_directory=os.getcwd(),
        weather_model_directory=os.path.join(
            os.getcwd(),
            'weather_files'
        ),
        output_projection='EPSG:4236',
    )

import os
from RAiDER.constants import _ZREF, _CUBE_SPACING_IN_M

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DEFAULT_DICT = AttributeDict(
    dict(
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
            cube_spacing_in_m=_CUBE_SPACING_IN_M,
            los_file=None,
            los_convention='isce',
            los_cube=None,
            orbit_file=None,
            verbose=True,
            raster_format='GTiff',
            file_format='GTiff',
            download_only=False,
            output_directory='.',
            weather_model_directory=None,
            output_projection='EPSG:4326',
            interpolate_time=True,
        )
    )

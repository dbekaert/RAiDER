import datetime
from RAiDER.delay import main

wetFilename   = '/Users/buzzanga/Software_InSAR/RAiDER-docs_git/notebooks/RAiDER_tutorial/GMAO_wet_20200103T000000_ztd.GTiff'
hydroFilename = '/Users/buzzanga/Software_InSAR/RAiDER-docs_git/notebooks/RAiDER_tutorial/GMAO_hydro_20200103T000000_ztd.GTiff'
dt = datetime.datetime(2020, 1, 3, 0, 0)
height_levels = [0, 50, 100, 500, 1000]
bounding_box  = [28, 39, -123, -112]
weather_model = 'GMAO'

main(dt, wetFilename, hydroFilename,
        bounding_box=bounding_box,
        weather_model=weather_model,
        height_levels=height_levels)

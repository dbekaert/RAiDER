
def writeWeatherLevelsToFile():
    '''
    Need to implement a separate file writing 
    routine for list-like outputs
    '''
    if outformat == 'hdf5':
        raise NotImplemented
    else:
        drv = gdal.GetDriverByName(outformat)
        hydro_ds = drv.Create(
            hydro_file_name, total_hydro.shape[2],
            total_hydro.shape[1], len(hts), gdal.GDT_Float64)
        for lvl, (hydro, ht) in enumerate(zip(total_hydro, hts), start=1):
            band = hydro_ds.GetRasterBand(lvl)
            band.SetDescription(str(ht))
            band.WriteArray(hydro)
        for f in set_geo_info:
            f(hydro_ds)
        hydro_ds = None
    
        wet_ds = drv.Create(
            wet_file_name, total_wet.shape[2],
            total_wet.shape[1], len(hts), gdal.GDT_Float64)
        for lvl, (wet, ht) in enumerate(zip(total_wet, hts), start=1):
            band = wet_ds.GetRasterBand(lvl)
            band.SetDescription(str(ht))
            band.WriteArray(wet)
        for f in set_geo_info:
            f(wet_ds)
        wet_ds = None


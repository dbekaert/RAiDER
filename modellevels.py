"""
Read model level WRF ECMWF file, maybe download it from the internet, idk.
"""

import numpy as np

def get_ecmwf(lat_min, lat_max, lat_step, lon_min, lon_max, lon_step):
    import ecmwfapi
    server = ecmwfapi.ECMWFDataServer()
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "expver": "1",
        "levelist": "all", # TODO: avoid the use of the value 'ALL'
        "levtype": "ml",
        "param": "r",
        "stream": "oper",
        "date": "2015-01-01",    #date: Specify a single date as "2015-08-01" or a period as "2015-08-01/to/2015-08-31".
        "type": "an",        #type: Use an (analysis) unless you have a particular reason to use fc (forecast).
        "time": "00:00:00",        #time: With type=an, time can be any of "00:00:00/06:00:00/12:00:00/18:00:00".  With type=fc, time can be any of "00:00:00/12:00:00",
        "step": "0",        #step: With type=an, step is always "0". With type=fc, step can be any of "3/6/9/12".
        "grid": f'{lat_step}/{lon_step}',    #grid: Only regular lat/lon grids are supported.
        "area": f'{lat_max}/{lon_min}/{lat_min}/{lon_max}',    #area: N/W/S/E, here we have Europe.
        "format": "netcdf",
        "target": "justr.nc",    #target: the name of the output file.
    })


# This function is copied (with slight modification) from
# https://confluence.ecmwf.int//display/CKB/ERA-Interim%3A+compute+geopotential+on+model+levels#ERA-Interim:computegeopotentialonmodellevels-Step2:Computegeopotentialonmodellevels.
# That script is licensed under the Apache license. I don't know if this
# is legal.
def calculategeoh(z, lnsp, ts, qs):
    heighttoreturn = np.zeros_like(ts)
    pressurelvs = np.zeros(ts.shape[0])

    Rd = 287.06

    z_h = 0

    #surface pressure
    sp = np.exp(lnsp)

    # A and B parameters to calculate pressures for model levels,
    #  extracted from an ECMWF ERA-Interim GRIB file and then hardcoded here
    pv =  [
      0.0000000000e+000, 2.0000000000e+001, 3.8425338745e+001, 6.3647796631e+001, 9.5636962891e+001,
      1.3448330688e+002, 1.8058435059e+002, 2.3477905273e+002, 2.9849584961e+002, 3.7397192383e+002,
      4.6461816406e+002, 5.7565112305e+002, 7.1321801758e+002, 8.8366040039e+002, 1.0948347168e+003,
      1.3564746094e+003, 1.6806403809e+003, 2.0822739258e+003, 2.5798886719e+003, 3.1964216309e+003,
      3.9602915039e+003, 4.9067070313e+003, 6.0180195313e+003, 7.3066328125e+003, 8.7650546875e+003,
      1.0376125000e+004, 1.2077445313e+004, 1.3775324219e+004, 1.5379804688e+004, 1.6819472656e+004,
      1.8045183594e+004, 1.9027695313e+004, 1.9755109375e+004, 2.0222203125e+004, 2.0429863281e+004,
      2.0384480469e+004, 2.0097402344e+004, 1.9584328125e+004, 1.8864750000e+004, 1.7961359375e+004,
      1.6899468750e+004, 1.5706449219e+004, 1.4411125000e+004, 1.3043218750e+004, 1.1632757813e+004,
      1.0209500000e+004, 8.8023554688e+003, 7.4388046875e+003, 6.1443164063e+003, 4.9417773438e+003,
      3.8509133301e+003, 2.8876965332e+003, 2.0637797852e+003, 1.3859125977e+003, 8.5536181641e+002,
      4.6733349609e+002, 2.1039389038e+002, 6.5889236450e+001, 7.3677425385e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      7.5823496445e-005, 4.6139489859e-004, 1.8151560798e-003, 5.0811171532e-003, 1.1142909527e-002,
      2.0677875727e-002, 3.4121163189e-002, 5.1690407097e-002, 7.3533833027e-002, 9.9674701691e-002,
      1.3002252579e-001, 1.6438430548e-001, 2.0247590542e-001, 2.4393314123e-001, 2.8832298517e-001,
      3.3515489101e-001, 3.8389211893e-001, 4.3396294117e-001, 4.8477154970e-001, 5.3570991755e-001,
      5.8616840839e-001, 6.3554745913e-001, 6.8326860666e-001, 7.2878581285e-001, 7.7159661055e-001,
      8.1125342846e-001, 8.4737491608e-001, 8.7965691090e-001, 9.0788388252e-001, 9.3194031715e-001,
      9.5182150602e-001, 9.6764522791e-001, 9.7966271639e-001, 9.8827010393e-001, 9.9401944876e-001,
      9.9763011932e-001, 1.0000000000e+000 ]
    levelSize = 60
    A = pv[0:levelSize+1]
    B = pv[levelSize+1:]

    Ph_levplusone = A[levelSize] + (B[levelSize]*sp)

    #Integrate up into the atmosphere from lowest level
    for lev, t_level, q_level in zip(range(levelSize, 0, -1), ts[::-1], qs[::-1]):
        #lev is the level number 1-60, we need a corresponding index into ts and qs
        ilevel = levelSize - lev

        #compute moist temperature
        t_level = t_level*(1 + 0.609133*q_level)

        #compute the pressures (on half-levels)
        Ph_lev = A[lev-1] + (B[lev-1] * sp)

        pressurelvs[ilevel] = Ph_lev

        if lev == 1:
            dlogP = np.log(Ph_levplusone/0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(Ph_levplusone/Ph_lev)
            dP    = Ph_levplusone - Ph_lev
            alpha = 1 - ((Ph_lev/dP)*dlogP)

        TRd = t_level*Rd

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the full level
        z_f = z_h + TRd*alpha

        #Convert geopotential to height 
        heighttoreturn[ilevel] = z_f / 9.80665

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h += TRd * dlogP

        Ph_levplusone = Ph_lev

    return heighttoreturn, pressurelvs


def load(tq_ml, lnspz):
    tq_ml = scipy.io.netcdf.netcdf_file(tq_ml, 'r', maskandscale=True)
    lnspz = scipy.io.netcdf.netcdf_file(lnspz, 'r', maskandscale=True)
    zs = lnspz.variables['z'][:].copy()
    lnsps = lnspz.variables['lnsp'][:].copy()
    ts = tq_ml.variables['t'][:].copy()
    qs = tq_ml.variables['qs'][:].copy()
    # TODO: modify calculategeoh to do it all at once
    for z, lnsp, t, q in zip(zs, lnsps, ts, qs):
        geo_ht, press = calculategeoh(z, lnsp, t, q)
        # TODO: find x and y
        weather = reader.import_grids('xs', 'ys', press, t, q, geo_ht, wrf.k1, wrf.k2, wrf.k3, projection='something', humidity_type='q')

#!/usr/bin/env python3
#
# Author: Jeremy Maurer, building on a prior code by Ray
# Copyright 2018


def makeVRT(filename, dtype = 'Float32'):
    '''
    Use an RSC file to create a GDAL-compatible VRT file for opening GACOS weather model files
    '''
    fields = readRSC(filename)
    string = vrtStr(fields['XMAX'],fields['YMAX'], fields['X_FIRST'],fields['Y_FIRST'], fields['X_STEP'],fields['Y_STEP'],filename.replace('.rsc', ''), dtype = dtype)
    writeStringToFile(string, filename.replace('.rsc', '').replace('.ztd', '') + '.vrt')
        

def writeStringToFile(string, filename):
    '''
    Write a string to a VRT file
    '''
    with open(filename, 'w') as f:
        f.write(string)
    

def readRSC(rscFilename):
    fields = {}
    with open(rscFilename, 'r') as f:
        for line in f:
            fieldName, value = line.strip().split()
            fields[fieldName] = value
    return fields
        

def vrtStr(xSize, ySize, lon1, lat1, lonStep, latStep, filename, dtype = 'Float32'):
    string = '''<VRTDataset rasterXSize="{xSize}" rasterYSize="{ySize}">
  <SRS>EPSG:4326</SRS>
  <GeoTransform> {lon1}, {lonStep},  0.0000000000000000e+00,  {lat1},  0.0000000000000000e+00, {latStep}</GeoTransform>
  <VRTRasterBand dataType="{dtype}" band="1" subClass="VRTRawRasterBand">
    <SourceFilename relativeToVRT="1">{filename}</SourceFilename>
  </VRTRasterBand>
</VRTDataset>
'''.format(xSize=xSize, ySize=ySize, filename=filename, dtype=dtype, lon1 = lon1, lat1 = lat1)

    return string


def convertAllFiles(dirLoc):
    '''
    convert all RSC files to VRT files contained in dirLoc
    '''
    import glob
    files = glob.glob('*.rsc')
    for f in files:
        makeVRT(f)


if __name__=='__main__': 
   import sys
   if len(sys.argv) == 2:
      makeVRT(sys.argv[1])
   elif len(sys.argv) == 3:
      convertAllFiles(sys.argv[1])
      print('Converting all RSC files in {}'.format(sys.argv[1]))
   else:
      print('Usage: ')
      print('python3 generateGACOSVRT.py <rsc_filename>')
      sys.exit(0)

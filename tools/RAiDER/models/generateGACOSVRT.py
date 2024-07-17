#
# Author: Jeremy Maurer, building on a prior code by Ray
# Copyright 2018


def makeVRT(filename, dtype='Float32') -> None:
    """Use an RSC file to create a GDAL-compatible VRT file for opening GACOS weather model files."""
    fields = readRSC(filename)
    string = vrtStr(
        fields['XMAX'],
        fields['YMAX'],
        fields['X_FIRST'],
        fields['Y_FIRST'],
        fields['X_STEP'],
        fields['Y_STEP'],
        filename.replace('.rsc', ''),
        dtype=dtype,
    )
    filename = filename.replace('.rsc', '').replace('.ztd', '') + '.vrt'
    with open(filename, 'w') as f:
        f.write(string)


def readRSC(rscFilename):
    fields = {}
    with open(rscFilename) as f:
        for line in f:
            fieldName, value = line.strip().split()
            fields[fieldName] = value
    return fields


def vrtStr(xSize, ySize, lon1, lat1, lonStep, latStep, filename, dtype='Float32'):
    return (
        f'<VRTDataset rasterXSize="{xSize}" rasterYSize="{ySize}">'
         '  <SRS>EPSG:4326</SRS>'
        f'  <GeoTransform> {lon1}, {lonStep},  0.0000000000000000e+00,  {lat1},  0.0000000000000000e+00, {latStep}</GeoTransform>'
        f'  <VRTRasterBand dataType="{dtype}" band="1" subClass="VRTRawRasterBand">'
        f'    <SourceFilename relativeToVRT="1">{filename}</SourceFilename>'
         '  </VRTRasterBand>'
         '</VRTDataset>'
    )


def convertAllFiles(dirLoc) -> None:
    """Convert all RSC files to VRT files contained in dirLoc."""
    import glob

    files = glob.glob('*.rsc')
    for f in files:
        makeVRT(f)


def main() -> None:
    import sys

    if len(sys.argv) == 2:
        makeVRT(sys.argv[1])
    elif len(sys.argv) == 3:
        convertAllFiles(sys.argv[1])
        print(f'Converting all RSC files in {sys.argv[1]}')
    else:
        print('Usage: ')
        print('python3 generateGACOSVRT.py <rsc_filename>')
        sys.exit(0)

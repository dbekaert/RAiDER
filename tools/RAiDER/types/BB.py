"""Types to help distinguish different bounding box formats."""

SNWE = tuple[float, float, float, float]
WSEN = tuple[float, float, float, float]  # used in dem_stitcher

SN = tuple[float, float]
WE = tuple[float, float]

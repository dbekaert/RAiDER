import os
import pytest
import requests
from unittest import mock

from test import TEST_DIR, pushd
from RAiDER.dem import download_dem
from RAiDER.gnss.downloadGNSSDelays import (
    check_url,read_text_file,in_box,fix_lons,get_ID,
    
)

# Test check_url with a valid and invalid URL
def test_check_url_valid():
  valid_url = "https://www.example.com/test.txt"
  with mock.patch.object(requests.Session, 'head') as mock_head:
    mock_head.return_value.status_code = 200  # Simulate successful response
    assert check_url(valid_url) == valid_url

def test_check_url_invalid():
  invalid_url = "https://www.not-a-real-website.com/notfound.txt"
  with mock.patch.object(requests.Session, 'head') as mock_head:
    mock_head.return_value.status_code = 404  # Simulate not found response
    assert check_url(invalid_url) == ''

# Test read_text_file with a valid and invalid filename
def test_read_text_file_valid():
  # Create a temporary test file with some content
  with open("test_file.txt", "w") as f:
    f.write("line1\nline2\n")
  try:
    lines = read_text_file("test_file.txt")
    assert lines == ["line1", "line2"]
  finally:
    os.remove("test_file.txt")  # Cleanup the temporary file

def test_read_text_file_invalid():
  invalid_filename = "not_a_file.txt"
  with pytest.raises(FileNotFoundError):
    read_text_file(invalid_filename)

# Test in_box with points inside and outside the box
def test_in_box_inside():
  lat = 38.0
  lon = -97.0
  llbox = [30, 40, -100, -90]  # Sample bounding box
  assert in_box(lat, lon, llbox) == True

def test_in_box_outside():
  lat = 50.0
  lon = -80.0
  llbox = [30, 40, -100, -90]  # Sample bounding box
  assert in_box(lat, lon, llbox) == False

# Test fix_lons with various longitudes
def test_fix_lons_positive():
  lon = 200.0
  assert fix_lons(lon) == -160.0

def test_fix_lons_negative():
  lon = -220.0
  assert fix_lons(lon) == 140.0

def test_fix_lons_positive_180():
  lon = 180.0
  assert fix_lons(lon) == 180.0

def test_fix_lons_negative_180():
  lon = -180.0
  assert fix_lons(lon) == -180.0

# Test get_ID with a valid line
def test_get_ID_valid():
  line = "ABCD 35.0 -98.0 100.0"
  stat_id, lat, lon, height = get_ID(line)
  assert stat_id == "ABCD"
  assert lat == 35.0
  assert lon == -98.0
  assert height == 100.0

# Test get_ID with an invalid line (not enough elements)
def test_get_ID_invalid():
  line = "ABCD 35.0"  # Missing longitude and height
  with pytest.raises(ValueError):
    get_ID(line)
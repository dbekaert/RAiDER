import logging
import pytest
import requests
from unittest import mock

from RAiDER.gnss.downloadGNSSDelays import (
    check_url,
    in_box,
    fix_lons,
    get_ID,
    download_UNR,
    main,
)

from test import pushd


# Test check_url with a valid and invalid URL
def test_check_url_valid():
    valid_url = "https://www.example.com/test.txt"
    with mock.patch.object(requests.Session, "head") as mock_head:
        mock_head.return_value.status_code = 200  # Simulate successful response
        assert check_url(valid_url) == valid_url


def test_check_url_invalid():
    invalid_url = "https://www.not-a-real-website.com/notfound.txt"
    with mock.patch.object(requests.Session, "head") as mock_head:
        mock_head.return_value.status_code = 404  # Simulate not found response
        assert check_url(invalid_url) == ""


# Test in_box with points inside and outside the box
def test_in_box_inside():
    lat = 38.0
    lon = -97.0
    llbox = [30, 40, -100, -90]  # Sample bounding box
    assert in_box(lat, lon, llbox)

def test_in_box_outside():
    lat = 50.0
    lon = -80.0
    llbox = [30, 40, -100, -90]  # Sample bounding box
    assert not in_box(lat, lon, llbox)

# Test fix_lons with various longitudes
def test_fix_lons_positive_to360():
    lon = 80.0
    assert fix_lons(lon, to360=True) == 260.0


def test_fix_lons_negative_to360():
    lon = -100.0
    assert fix_lons(lon, to360=True) == 80.0


def test_fix_lons_0_to180():
    lon = 0.0
    assert fix_lons(lon) == -180.0


def test_fix_lons_360_to180():
    lon = 360.0
    assert fix_lons(lon) == 180.0


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


def test_download_UNR(tmp_path):
    expected_path = (
        "https://geodesy.unr.edu/gps_timeseries/IGS20/trop/MORZ/"
        "MORZ.2020.trop.zip"
    )
    statID = "MORZ"
    year = 2020
    with pushd(tmp_path):
        outDict = download_UNR(statID, year)
        assert outDict["path"] == expected_path


def test_download_UNR_2(caplog):
    statID = "MORZ"
    year = 2000
    
    # Capture logs at the WARNING level and above
    with caplog.at_level(logging.WARNING):
        result = download_UNR(statID, year, download=True)
    
    # 1. Assert the correct warning was logged
    expected_warning = f"Skipping {statID}: Not found in either archive for {year}."
    assert expected_warning in caplog.text
    
    # 2. Assert the function returns the expected dictionary with a falsy path
    assert result["ID"] == statID
    assert result["year"] == year
    assert not result["path"]  # Asserts path is None, False, or empty string


def test_download_UNR_3(caplog):
    statID = "DUMY"
    year = 2020
    
    with caplog.at_level(logging.WARNING):
        result = download_UNR(statID, year, download=True)
        
    expected_warning = f"Skipping {statID}: Not found in either archive for {year}."
    assert expected_warning in caplog.text
    
    assert result["ID"] == statID
    assert result["year"] == year
    assert not result["path"]


def test_download_UNR_4():
    statID = "MORZ"
    year = 2020
    with pytest.raises(NotImplementedError):
        download_UNR(statID, year, baseURL="www.google.com")


@pytest.mark.skip
def test_main():
    iargs = None
    main(inps=iargs)

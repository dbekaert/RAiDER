from argparse import ArgumentParser
from datetime import datetime, time, date

import os
import pytest

import numpy as np

from test import TEST_DIR

from RAiDER.cli.args import DateGroupUnparsed, LOSGroupUnparsed, TimeGroup
from RAiDER.cli.validators import (
    getBufferedExtent, isOutside, isInside,
    coerce_into_date,
    parse_bbox, parse_dates, parse_weather_model, get_los
)

SCENARIO = os.path.join(TEST_DIR, "scenario_4")

@pytest.fixture
def parser():
    return ArgumentParser()


@pytest.fixture
def llsimple():
    lats = (10, 12)
    lons = (-72, -74)
    return lats, lons


@pytest.fixture
def latwrong():
    lats = (12, 10)
    lons = (-72, -74)
    return lats, lons


@pytest.fixture
def lonwrong():
    lats = (10, 12)
    lons = (-72, -74)
    return lats, lons


@pytest.fixture
def llarray():
    lats = np.arange(10, 12.1, 0.1)
    lons = np.arange(-74, -71.9, 0.2)
    return lats, lons


@pytest.fixture
def args1():
    test_file = os.path.join(SCENARIO, 'los.rdr')
    args = {
        'los_file': test_file,
        'los_convention': 'isce',
        'ray_trace': False,
    }
    return args



def test_enforce_wm():
    with pytest.raises(NotImplementedError):
        parse_weather_model('notamodel', 'fakeaoi')


def test_get_los_ray(args1):
    args = args1
    los_group_unparsed = LOSGroupUnparsed(**args)
    los = get_los(los_group_unparsed)
    assert not los.ray_trace()
    assert los.is_Projected()


def test_date_type():
    assert coerce_into_date("2020-10-1") == date(2020, 10, 1)
    assert coerce_into_date("2020101") == date(2020, 10, 1)

    with pytest.raises(ValueError):
        coerce_into_date("foobar")


@pytest.mark.parametrize("input,expected", (
    ("T23:00:01.000000", time(23, 0, 1)),
    ("T23:00:01.000000", time(23, 0, 1)),
    ("T230001.000000", time(23, 0, 1)),
    ("230001.000000", time(23, 0, 1)),
    ("T23:00:01", time(23, 0, 1)),
    ("23:00:01", time(23, 0, 1)),
    ("T230001", time(23, 0, 1)),
    ("230001", time(23, 0, 1)),
    ("T23:00", time(23, 0, 0)),
    ("T2300", time(23, 0, 0)),
    ("23:00", time(23, 0, 0)),
    ("2300", time(23, 0, 0))
))
@pytest.mark.parametrize("timezone", ("", "z", "+0000"))
def test_time_type(input, timezone, expected):
    assert TimeGroup.coerce_into_time(input + timezone) == expected


def test_time_type_error():
    with pytest.raises(ValueError):
        TimeGroup.coerce_into_time("foobar")


def test_date_list_action():
    date_group_unparsed = DateGroupUnparsed(
        date_start='20200101',
    )
    assert coerce_into_date(date_group_unparsed.date_start) == date(2020,1,1)
    assert parse_dates(date_group_unparsed).date_list == [date(2020,1,1)]

    date_group_unparsed.date_end = '20200103'
    assert coerce_into_date(date_group_unparsed.date_end) == date(2020,1,3)
    assert parse_dates(date_group_unparsed).date_list == [date(2020,1,1), date(2020,1,2), date(2020,1,3)]

    date_group_unparsed.date_end = '20200112'
    date_group_unparsed.date_step = '5'
    assert parse_dates(date_group_unparsed).date_list == [date(2020,1,1), date(2020,1,6), date(2020,1,11)]


def test_bbox_action():
    bbox_str = "45 46 -72 -70"
    assert len(parse_bbox(bbox_str)) == 4

    assert parse_bbox(bbox_str) == (45, 46, -72, -70)

    with pytest.raises(ValueError):
        parse_bbox("20 20 30 30")
    with pytest.raises(ValueError):
        parse_bbox("30 100 20 40")
    with pytest.raises(ValueError):
        parse_bbox("10 30 40 190")


def test_ll1(llsimple):
    lats, lons = llsimple
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_ll2(latwrong):
    lats, lons = latwrong
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_ll3(lonwrong):
    lats, lons = lonwrong
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_ll4(llarray):
    lats, lons = llarray
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_isOutside1(llsimple):
    extent1 = getBufferedExtent(*llsimple)
    extent2 = extent1[0] + 1, extent1[1] + 1, extent1[2] + 1, extent1[3] + 1
    assert isOutside(extent1, extent2)


def test_isOutside2(llsimple):
    extent = getBufferedExtent(*llsimple)
    assert not isOutside(extent, extent)


def test_isInside(llsimple):
    extent1 = getBufferedExtent(*llsimple)
    extent2 = extent1[0] + 1, extent1[1] + 1, extent1[2] + 1, extent1[3] + 1
    assert isInside(extent1, extent1)
    assert not isInside(extent1, extent2)

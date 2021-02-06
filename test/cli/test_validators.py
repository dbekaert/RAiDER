from argparse import ArgumentParser, ArgumentTypeError
from datetime import date, time

import pytest

from RAiDER.cli.validators import (
    BBoxAction, DateListAction, IntegerMappingType, IntegerType, MappingType,
    date_type, time_type
)


@pytest.fixture
def parser():
    return ArgumentParser()


def test_mapping_type_default():
    mapping = MappingType(foo=42, bar="baz").default(None)

    assert mapping("foo") == 42
    assert mapping("bar") == "baz"
    assert mapping("hello") is None


def test_mapping_type_no_default():
    mapping = MappingType(foo=42, bar="baz")

    assert mapping("foo") == 42
    assert mapping("bar") == "baz"
    with pytest.raises(KeyError):
        assert mapping("hello")


def test_integer_type():
    integer = IntegerType(0, 100)

    assert integer("0") == 0
    assert integer("100") == 100
    with pytest.raises(ArgumentTypeError):
        integer("-10")
    with pytest.raises(ArgumentTypeError):
        integer("101")


def test_integer_mapping_type_default():
    integer = IntegerMappingType(0, 100, random=42).default(-1)

    assert integer("0") == 0
    assert integer("100") == 100
    assert integer("random") == 42
    assert integer("foo") == -1
    with pytest.raises(ArgumentTypeError):
        integer("-1")


def test_integer_mapping_type_no_default():
    integer = IntegerMappingType(0, 100, random=42)

    assert integer("0") == 0
    assert integer("100") == 100
    assert integer("random") == 42
    with pytest.raises(KeyError):
        integer("foo")
    with pytest.raises(ArgumentTypeError):
        integer("-1")


def test_date_type():
    assert date_type("2020-10-1") == date(2020, 10, 1)
    assert date_type("2020101") == date(2020, 10, 1)

    with pytest.raises(ArgumentTypeError):
        date_type("foobar")


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
    assert time_type(input + timezone) == expected


def test_time_type_error():
    with pytest.raises(ArgumentTypeError):
        time_type("foobar")


def test_date_list_action(parser):
    # DateListAction must be used with type=date_type
    with pytest.raises(ValueError):
        parser.add_argument(
            "--datelist",
            action=DateListAction,
        )

    parser.add_argument(
        "--datelist",
        action=DateListAction,
        nargs="+",
        type=date_type,
    )

    args = parser.parse_args(["--datelist", "2020-1-1"])
    assert args.datelist == [date(2020, 1, 1)]

    args = parser.parse_args(["--datelist", "2020-1-1", "2020-1-3"])
    assert args.datelist == [
        date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(["--datelist", "2020-1-1", "2020-1-2", "2.5"])


def test_bbox_action(parser):
    # BBoxAction must be used with nargs=4
    with pytest.raises(ValueError):
        parser.add_argument(
            "--bbox",
            action=BBoxAction,
        )

    parser.add_argument(
        "--bbox_float",
        action=BBoxAction,
        nargs=4,
        type=float,
    )

    parser.add_argument(
        "--bbox_int",
        action=BBoxAction,
        nargs=4,
        type=int,
    )

    args = parser.parse_args(["--bbox_float", "10", "20", "30", "40"])
    assert args.bbox_float == [10., 20., 30., 40.]
    args = parser.parse_args(["--bbox_int", "10", "20", "30", "40"])
    assert args.bbox_int == [10, 20, 30, 40]

    with pytest.raises(SystemExit):
        parser.parse_args(["--bbox_int", "10", "10", "20", "20"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--bbox_int", "30", "100", "20", "40"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--bbox_int", "10", "30", "40", "190"])

import datetime
import pytest

import numpy as np

from RAiDER.mathFcns import (
    _least_nonzero, 
    cosd, 
    sind, 
    round_date,
    round_time,
    robmin,
    robmax,
    padLower,
)


@pytest.fixture
def test_arr():
    test = np.array(
            [
                [
                    [np.nan, 1, 2], 
                    [np.nan, np.nan, 1],
                ],
                [
                    [1, 2, 3],
                    [2, 3, 4],
                ],
                [
                    [3, 4, 5],
                    [4, 5, 6]
                ]
            ]
        )
    return test

@pytest.fixture
def true_arr():
    a_true = np.array(
            [
                [1, 1],
                [1, 2],
                [3, 4]
            ]
        )
    return a_true


def test_sind():
    assert np.allclose(
        sind(np.array([0, 30, 90, 180])),
        np.array([0, 0.5, 1, 0])
    )


def test_cosd():
    assert np.allclose(
        cosd(np.array([0, 60, 90, 180])),
        np.array([1, 0.5, 0, -1])
    )


def test_least_nonzero():
    a = np.arange(20, dtype="float64").reshape(2, 2, 5)
    a[0, 0, 0] = np.nan
    a[1, 1, 0] = np.nan

    assert np.allclose(
        _least_nonzero(a),
        np.array([
            [1, 5],
            [10, 16]
        ]),
        atol=1e-16
    )


def test_least_nonzero_2():
    a = np.array([
        [[10., 5., np.nan],
         [11., np.nan, 1],
         [18, 17, 16]],

        [[np.nan, 12., 6.],
         [np.nan, 13., 20.],
         [np.nan, np.nan, np.nan]]
    ])

    assert np.allclose(
        _least_nonzero(a),
        np.array([
            [10, 11, 18],
            [12, 13, np.nan]
        ]),
        atol=1e-16,
        equal_nan=True
    )


def test_round_time_1():
    assert round_time(
            datetime.datetime(2020, 1, 1, 1, 2, 35), 
        ) == datetime.datetime(2020, 1, 1, 1, 3, 0)

def test_round_time_2():
    assert round_time(
            datetime.datetime(2020, 1, 1, 1, 2, 35), 
            roundTo = 30
        ) == datetime.datetime(2020, 1, 1, 1, 2, 30)

def test_round_time_3():
    assert round_time(
            datetime.datetime(2020, 1, 1, 1, 2, 35), 
            roundTo = 15
        ) == datetime.datetime(2020, 1, 1, 1, 2, 30)

def test_round_time_4():
    assert round_time(
            datetime.datetime(2020, 1, 1, 1, 2, 30), 
        ) == datetime.datetime(2020, 1, 1, 1, 3, 0)

def test_round_date_1():
    assert round_date(
            datetime.datetime(2020, 1, 15, 5, 5, 4), 
            datetime.timedelta(hours=6)
        ) == datetime.datetime(2020, 1, 15, 6, 0, 0)

def test_round_date_2():
    assert round_date(
            datetime.datetime(2020, 1, 15, 0, 0, ), 
            datetime.timedelta(hours=6)
        ) == datetime.datetime(2020, 1, 15, 0, 0, 0)

def test_round_date_3():
    assert round_date(
            datetime.datetime(2020, 1, 15), 
            datetime.timedelta(hours=6)
        ) == datetime.datetime(2020, 1, 15, 0, 0, 0)

def test_round_date_4():
    assert round_date(
            datetime.datetime(2020, 2, 29, 23, 55, 0), 
            datetime.timedelta(hours=6)
        ) == datetime.datetime(2020, 3, 1, 0, 0, 0)

def test_round_date_5():
    assert round_date(
            datetime.datetime(2020, 2, 29, 23, 30, 0), 
            datetime.timedelta(hours=1)
        ) == datetime.datetime(2020, 3, 1, 0, 0, 0)

def test_robmin_1():
    assert robmin(5.1) == 5.1
    assert robmin([5.1]) == [5.1]
    assert robmin(np.array([5.1])) == [5.1]

def test_robmin_2():
    assert robmin([]) == 'N/A'
    assert robmin(np.array([])) == 'N/A'

def test_robmin_3():
    assert robmin([1, 2, 3, 4]) == 1
    assert robmin(np.array([1, 2, 3, 4])) == 1

def test_robmax_1():
    assert robmax(5.1) == 5.1
    assert robmax([5.1]) == [5.1]
    assert robmax(np.array([5.1])) == [5.1]

def test_robmax_2():
    assert robmax([]) == 'N/A'
    assert robmax(np.array([])) == 'N/A'

def test_robmax_3():
    assert robmax([1, 2, 3, 4]) == 4
    assert robmax(np.array([1, 2, 3, 4])) == 4

def test_least_nonzero():
    test = np.random.randn(2, 4)
    test[0, 0] = np.nan
    test[1, 0] = np.nan
    test[1, 1] = np.nan
    a = _least_nonzero(test)
    assert a[0] == test[0,1]
    assert a[1] == test[1,2]

def test_least_nonzero_2(test_arr, true_arr):
    a_true = true_arr
    a = _least_nonzero(test_arr)
    assert np.allclose(a, a_true)

def test_padLower(test_arr, true_arr):
    true = np.concatenate(
            (true_arr[...,np.newaxis], test_arr), 
            axis=2
        ) 
    out = padLower(test_arr)
    assert np.allclose(out, true, equal_nan = True)

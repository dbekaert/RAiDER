import os
import pytest

import numpy as np

from test import DATA_DIR, pushd

from RAiDER.delay import (
    checkQueryPntsFile
)


def test_cqpf1():
    assert checkQueryPntsFile('does_not_exist.h5', None)

import os
from contextlib import contextmanager
from pathlib import Path

test_dir = Path(__file__).parents[0]

@contextmanager
def pushd(dir):
    """
    Change the current working directory within a context.
    """
    prevdir = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(prevdir)


TEST_DIR = test_dir.absolute()
DATA_DIR = os.path.join(TEST_DIR, "data")
GEOM_DIR = os.path.join(TEST_DIR, 'test_geom')

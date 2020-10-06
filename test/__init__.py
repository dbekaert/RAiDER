import inspect
import os
from contextlib import contextmanager


def get_test_dir():
    """
    Return the absolute path to the test directory.
    """
    source_file = inspect.getsourcefile(lambda: None)
    if os.path.isfile(source_file):
        return os.path.dirname(os.path.abspath(source_file))

    return os.getcwd()


@contextmanager
def pushd(dir):
    """
    Change the current working directory within a context.
    """
    prevdir = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(prevdir)


TEST_DIR = get_test_dir()
DATA_DIR = os.path.join(TEST_DIR, "data")
GEOM_DIR = os.path.join(TEST_DIR, 'test_geom')

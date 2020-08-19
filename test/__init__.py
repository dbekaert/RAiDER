import inspect
import os
import pathlib
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
def pushd(directory):
    """
    Change the current working directory within a context.
    """
    prevdir = os.getcwd()
    os.chdir(directory)
    yield
    os.chdir(prevdir)


TEST_DIR = pathlib.Path(get_test_dir())
DATA_DIR = TEST_DIR / "data"

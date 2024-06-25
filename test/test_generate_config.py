'''
Tests the functionality of `raider.py --generate_config <example_file>`.
For any given example file name, the script should produce that run
configuration file in the current working directory, along wihh any
corresponding data files.
If such a file already exists, the script should prompt the user to
confirm overwriting the file.
'''
import os
import subprocess
import tempfile
from contextlib import contextmanager


@contextmanager
def cd_to_temp_dir():
    old_pwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            yield
    finally:
        os.chdir(old_pwd)


def test_template():
    with cd_to_temp_dir():
        subprocess.run(['raider.py', '--generate_config', 'template'])
        assert os.path.exists('template.yaml')


def test_example_la_bbox():
    with cd_to_temp_dir():
        subprocess.run(['raider.py', '--generate_config', 'example_LA_bbox'])
        assert os.path.exists('example_LA_bbox.yaml')


def test_example_la_gnss():
    with cd_to_temp_dir():
        subprocess.run(['raider.py', '--generate_config', 'example_LA_GNSS'])
        assert os.path.exists('example_LA_GNSS.yaml')
        assert os.path.exists('example_LA_GNSS.csv')


def test_example_uk_isce():
    with cd_to_temp_dir():
        subprocess.run(['raider.py', '--generate_config', 'example_UK_isce'])
        assert os.path.exists('example_UK_isce.yaml')
        assert os.path.exists('example_UK_isce-S1B_OPER_AUX_POEORB_OPOD_20211122T112354_V20211101T225942_20211103T005942.EOF')


def test_confirm_overwrite_yes():
    with cd_to_temp_dir():
        with open('template.yaml', 'w') as f:
            f.write('overwrite me')
        process = subprocess.Popen(
            ['raider.py', '--generate_config', 'template'],
            stdin=subprocess.PIPE,
        )
        process.stdin.write(b'y\n')
        process.stdin.close()
        process.wait()
        assert os.path.exists('template.yaml')
        with open('template.yaml') as f:
            assert f.read() != 'overwrite me'


def test_confirm_overwrite_no():
    with cd_to_temp_dir():
        with open('template.yaml', 'w') as f:
            f.write("don't overwrite me")
        process = subprocess.Popen(
            ['raider.py', '--generate_config', 'template'],
            stdin=subprocess.PIPE,
        )
        process.stdin.write(b'n\n')
        process.stdin.close()
        process.wait()
        assert os.path.exists('template.yaml')
        with open('template.yaml') as f:
            assert f.read() == "don't overwrite me"

import os
import subprocess

from test import TEST_DIR

def test_scenario_1():
    test_path = os.path.join(TEST_DIR, "scenario_1", 'raider_example_1.yaml')
    process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True,)
    assert process.returncode == 0

    # Clean up files
    subprocess.run(['rm', '-f', './HRRR*'])
    subprocess.run(['rm', '-rf', './weather_files'])


def test_scenario_2():
    test_path = os.path.join(TEST_DIR, "scenario_2", 'raider_example_2.yaml')
    process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True)
    assert process.returncode == 0

    # Clean up files
    subprocess.run(['rm', '-f', './HRRR*'])
    subprocess.run(['rm', '-rf', './weather_files'])


def test_scenario_3():
    test_path = os.path.join(TEST_DIR, "scenario_3", 'raider_example_3.yaml')
    process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True)
    assert process.returncode == 0

    # Clean up files
    subprocess.run(['rm', '-f', './HRRR*'])
    subprocess.run(['rm', '-rf', './weather_files'])

def test_scenario_4():
    test_path = os.path.join(TEST_DIR, "scenario_3", 'raider_example_4.yaml')
    process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True)
    assert process.returncode == 0

    # Clean up files
    subprocess.run(['rm', '-f', './HRRR*'])
    subprocess.run(['rm', '-rf', './weather_files'])


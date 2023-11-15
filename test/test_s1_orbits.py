import netrc
from pathlib import Path

import pytest

from RAiDER import s1_orbits


def test__ensure_orbit_credentials(monkeypatch):
    class EmptyNetrc():
        def __init__(self, netrc_file):
            self.netrc_file = netrc_file
            self.hosts = {}
        def __str__(self):
            return str(self.hosts)

    # No .netrc, no ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', EmptyNetrc, raising=False)
        mp.delenv('ESA_USERNAME', raising=False)
        mp.delenv('ESA_PASSWORD', raising=False)
        with pytest.raises(ValueError):
            s1_orbits._ensure_orbit_credentials()

    # No .netrc, set ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', EmptyNetrc, raising=False)
        mp.setenv('ESA_USERNAME', 'foo')
        mp.setenv('ESA_PASSWORD', 'bar')
        mp.setattr(Path, 'write_text', lambda self, write_text: write_text)
        written_credentials = s1_orbits._ensure_orbit_credentials()
        assert written_credentials == str({s1_orbits.ESA_CDSE_HOST: ('foo', None, 'bar')})

    class NoCDSENetrc():
        def __init__(self, netrc_file):
            self.netrc_file = netrc_file
            self.hosts = {'fizz.buzz.org': ('foo', None, 'bar')}
        def __str__(self):
            return str(self.hosts)

    # No CDSE in .netrc, no ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', NoCDSENetrc, raising=False)
        mp.delenv('ESA_USERNAME', raising=False)
        mp.delenv('ESA_PASSWORD', raising=False)
        with pytest.raises(ValueError):
            s1_orbits._ensure_orbit_credentials()

    # No CDSE in .netrc, set ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', NoCDSENetrc, raising=False)
        mp.setenv('ESA_USERNAME', 'foo')
        mp.setenv('ESA_PASSWORD', 'bar')
        mp.setattr(Path, 'write_text', lambda self, write_text: write_text)
        written_credentials = s1_orbits._ensure_orbit_credentials()
        assert written_credentials == str({'fizz.buzz.org': ('foo', None, 'bar'), s1_orbits.ESA_CDSE_HOST: ('foo', None, 'bar')})

    class CDSENetrc():
        def __init__(self, netrc_file):
            self.netrc_file = netrc_file
            self.hosts = {s1_orbits.ESA_CDSE_HOST: ('foo', None, 'bar')}
        def __str__(self):
            return str(self.hosts)

    # cdse in .netrc, no ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', CDSENetrc, raising=False)
        mp.delenv('ESA_USERNAME', raising=False)
        mp.delenv('ESA_PASSWORD', raising=False)
        written_credentials = s1_orbits._ensure_orbit_credentials()
        assert written_credentials is None

    # cdse in .netrc, set ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', CDSENetrc, raising=False)
        mp.setenv('ESA_USERNAME', 'foo')
        mp.setenv('ESA_PASSWORD', 'bar')
        written_credentials = s1_orbits._ensure_orbit_credentials()
        assert written_credentials is None


def test_get_esa_cse_credentials(monkeypatch):
    class CDSENetrc():
        def __init__(self, netrc_file):
            self.netrc_file = netrc_file
            self.hosts = {s1_orbits.ESA_CDSE_HOST: ('foo', None, 'bar')}
        def __str__(self):
            return str(self.hosts)

    # cdse in .netrc, no ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', CDSENetrc, raising=False)
        username, password = s1_orbits.get_esa_cdse_credentials()

    assert username == 'foo'
    assert password == 'bar'

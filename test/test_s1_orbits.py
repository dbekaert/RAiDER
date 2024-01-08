import netrc
from pathlib import Path

import eof.download
import pytest

from RAiDER import s1_orbits


def test_ensure_orbit_credentials(monkeypatch):
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
            s1_orbits.ensure_orbit_credentials()

    # No .netrc, set ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', EmptyNetrc, raising=False)
        mp.setenv('ESA_USERNAME', 'foo')
        mp.setenv('ESA_PASSWORD', 'bar')
        mp.setattr(Path, 'write_text', lambda self, write_text: write_text)
        written_credentials = s1_orbits.ensure_orbit_credentials()
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
            s1_orbits.ensure_orbit_credentials()

    # No CDSE in .netrc, set ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', NoCDSENetrc, raising=False)
        mp.setenv('ESA_USERNAME', 'foo')
        mp.setenv('ESA_PASSWORD', 'bar')
        mp.setattr(Path, 'write_text', lambda self, write_text: write_text)
        written_credentials = s1_orbits.ensure_orbit_credentials()
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
        written_credentials = s1_orbits.ensure_orbit_credentials()
        assert written_credentials is None

    # cdse in .netrc, set ESA CDSE env variables
    with monkeypatch.context() as mp:
        mp.setattr(netrc, 'netrc', CDSENetrc, raising=False)
        mp.setenv('ESA_USERNAME', 'foo')
        mp.setenv('ESA_PASSWORD', 'bar')
        written_credentials = s1_orbits.ensure_orbit_credentials()
        assert written_credentials is None


def test_get_orbits_from_slc_ids(mocker):
    side_effect = [
        [Path('foo.txt')],
        [Path('bar.txt'), Path('fiz.txt')],
    ]
    mocker.patch('eof.download.download_eofs',
                 side_effect=side_effect)

    orbit_files = s1_orbits.get_orbits_from_slc_ids(
        ['S1A_IW_SLC__1SSV_20150621T120220_20150621T120232_006471_008934_72D8']
    )
    assert orbit_files == [Path('foo.txt')]
    assert eof.download.download_eofs.call_count == 1
    eof.download.download_eofs.assert_called_with(
        [ '20150621T120220'], ['S1A'], save_dir=str(Path.cwd())
    )

    orbit_files = s1_orbits.get_orbits_from_slc_ids(
        ['S1B_IW_SLC__1SDV_20201115T162313_20201115T162340_024278_02E29D_5C54',
         'S1A_IW_SLC__1SDV_20201203T162353_20201203T162420_035524_042744_6D5C']
    )
    assert orbit_files == [Path('bar.txt'), Path('fiz.txt')]
    assert eof.download.download_eofs.call_count == 2
    eof.download.download_eofs.assert_called_with(
        ['20201115T162313', '20201203T162353'],
        ['S1B', 'S1A'],
        save_dir=str(Path.cwd())
    )

import netrc
import os
import re
from pathlib import Path
from platform import system
from typing import List, Optional

import eof.download
from hyp3lib import get_orb
from RAiDER.logger import logger


ESA_CDSE_HOST = 'dataspace.copernicus.eu'
NASA_EDL_HOST = 'urs.earthdata.nasa.gov'


def _netrc_path() -> Path:
    netrc_name = '_netrc' if system().lower() == 'windows' else '.netrc'
    return Path.home() / netrc_name


def ensure_orbit_credentials() -> Optional[int]:
    """Ensure credentials exist for ESA's CDSE and ASF's S1QC to download orbits

    This method will prefer to use CDSE and NASA Earthdata credentials from your `~/.netrc` file if they exist,
    otherwise will look for environment variables and update or create your `~/.netrc` file. The environment variables
    used are:
        CDSE: ESA_USERNAME, ESA_PASSWORD
        S1QC: EARTHDATA_USERNAME, EARTHDATA_PASSWORD

     Returns `None` if the `~/.netrc` file did not need to be updated and the number of characters written if it did.
    """
    netrc_file = _netrc_path()

    # netrc needs a netrc file; if missing create an empty one.
    if not netrc_file.exists():
        netrc_file.touch(mode=0o600)

    netrc_credentials = netrc.netrc(netrc_file)
    if ESA_CDSE_HOST in netrc_credentials.hosts and NASA_EDL_HOST in netrc_credentials.hosts:
        return

    if ESA_CDSE_HOST not in netrc_credentials.hosts:
        username = os.environ.get('ESA_USERNAME')
        password = os.environ.get('ESA_PASSWORD')
        if username is None or password is None:
            raise ValueError('Credentials are required for fetching orbit data from dataspace.copernicus.eu!\n'
                             'Either add your credentials to ~/.netrc or set the ESA_USERNAME and ESA_PASSWORD '
                             'environment variables.')

        netrc_credentials.hosts[ESA_CDSE_HOST] = (username, None, password)

    if NASA_EDL_HOST not in netrc_credentials.hosts:
        username = os.environ.get('EARTHDATA_USERNAME')
        password = os.environ.get('EARTHDATA_PASSWORD')
        if username is None or password is None:
            raise ValueError(f'Credentials are required for fetching orbit data from s1qc.asf.alaska.edu!\n'
                             'Either add your credentials to ~/.netrc or set the EARTHDATA_USERNAME and'
                             ' EARTHDATA_PASSWORD environment variables.')

        netrc_credentials.hosts[NASA_EDL_HOST] = (username, None, password)

    return netrc_file.write_text(str(netrc_credentials))


def get_orbits_from_slc_ids(slc_ids: List[str], directory=Path.cwd()) -> List[Path]:
    """Download all orbit files for a set of SLCs

    This method will ensure that the downloaded orbit files cover the entire acquisition start->stop time

    Returns a list of orbit file paths
    """
    missions = [slc_id[0:3] for slc_id in slc_ids]
    start_times = [re.split(r'_+', slc_id)[4] for slc_id in slc_ids]
    stop_times = [re.split(r'_+', slc_id)[5] for slc_id in slc_ids]

    orb_files = download_eofs(start_times + stop_times, missions * 2, str(directory))

    return orb_files


def get_orbits_from_slc_ids_hyp3lib(
    slc_ids: list, orbit_directory: str = None
) -> dict:
    """Reference: https://github.com/ACCESS-Cloud-Based-InSAR/DockerizedTopsApp/blob/dev/isce2_topsapp/localize_orbits.py#L23"""

    # Populates env variables to netrc as required for sentineleof
    _ = ensure_orbit_credentials()
    esa_username, _, esa_password = netrc.netrc().authenticators(ESA_CDSE_HOST)
    esa_credentials = esa_username, esa_password

    orbit_directory = orbit_directory or 'orbits'
    orbit_dir = Path(orbit_directory)
    orbit_dir.mkdir(exist_ok=True)

    orbit_fetcher =  get_orb.downloadSentinelOrbitFile

    orbits = []
    for scene in slc_ids:
        orbit_file, _ = orbit_fetcher(scene, str(orbit_dir), esa_credentials=esa_credentials, providers=('ASF', 'ESA'))
        orbits.append(orbit_file)

    orbits = sorted(list(set(orbits)))

    return orbits


def download_eofs(dts: list, missions: list, save_dir: str):
    """Wrapper around sentineleof to first try downloading from ASF and fall back to CDSE"""
    _ = ensure_orbit_credentials()

    orb_files = []
    for dt, mission in zip(dts, missions):
        dt = dt if isinstance(dt, list) else [dt]
        mission = mission if isinstance(mission, list) else [mission]

        try:
            orb_file = eof.download.download_eofs(dt, mission, save_dir=save_dir, force_asf=True)
        except:
            logger.error(f'Could not download orbit from ASF, trying ESA...')
            orb_file = eof.download.download_eofs(dt, mission, save_dir=save_dir, force_asf=False)

        orb_file = orb_file[0] if isinstance(orb_file, list) else orb_file
        orb_files.append(orb_file)

    if not len(orb_files) == len(dts):
        raise Exception(f'Missing {len(dts) - len(orb_files)} orbit files! dts={dts}, orb_files={len(orb_files)}')

    return orb_files

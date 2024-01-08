import netrc
import os
import re
from pathlib import Path
from platform import system
from typing import List, Optional, Tuple

import eof.download


ESA_CDSE_HOST = 'dataspace.copernicus.eu'


def _netrc_path() -> Path:
    netrc_name = '_netrc' if system().lower() == 'windows' else '.netrc'
    return Path.home() / netrc_name


def ensure_orbit_credentials() -> Optional[int]:
    """Ensure credentials exist for ESA's CDSE to download orbits

    This method will prefer to use CDSE credentials from your `~/.netrc` file if they exist,
    otherwise will look for ESA_USERNAME and ESA_PASSWORD environment variables and
     update or create your `~/.netrc` file.

     Returns `None` if the `~/.netrc` file did not need to be updated and the number of characters written if it did.
    """
    netrc_file = _netrc_path()

    # netrc needs a netrc file; if missing create an empty one.
    if not netrc_file.exists():
        netrc_file.touch(mode=0o600)

    netrc_credentials = netrc.netrc(netrc_file)
    if ESA_CDSE_HOST in netrc_credentials.hosts:
        return

    username = os.environ.get('ESA_USERNAME')
    password = os.environ.get('ESA_PASSWORD')
    if username is None and password is None:
        raise ValueError('Credentials are required for fetching orbit data from dataspace.copernicus.eu!\n'
                         'Either add your credentials to ~/.netrc or set the ESA_USERNAME and ESA_PASSWORD '
                         'environment variables.')

    netrc_credentials.hosts[ESA_CDSE_HOST] = (username, None, password)
    return netrc_file.write_text(str(netrc_credentials))


def get_orbits_from_slc_ids(slc_ids: List[str], directory=Path.cwd()) -> List[Path]:
    """Download all orbit files for a set of SLCs

    This method will ensure that the downloaded orbit files cover the entire acquisition start->stop time

    Returns a list of orbit file paths
    """
    _ = ensure_orbit_credentials()

    missions = [slc_id[0:3] for slc_id in slc_ids]
    start_times = [re.split(r'_+', slc_id)[4] for slc_id in slc_ids]
    stop_times = [re.split(r'_+', slc_id)[5] for slc_id in slc_ids]
    
    orb_files = eof.download.download_eofs(start_times + stop_times, missions * 2, save_dir=str(directory))

    return orb_files

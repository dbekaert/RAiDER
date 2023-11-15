import netrc
import os
from pathlib import Path
from platform import system
from typing import Optional, Tuple


ESA_CDSE_HOST = 'dataspace.copernicus.eu'


def _netrc_path() -> Path:
    netrc_name = '_netrc' if system().lower() == 'windows' else '.netrc'
    return Path.home() / netrc_name


def _ensure_orbit_credential() -> Optional[int]:
    """Ensure credentials exist for ESA's CDSE to download orbits

    This method will prefer to use CDSE credentials from your `~/.netrc` file if they exist,
    otherwise will look for ESA_USERNAME and ESA_PASSWORD environment variables and
     update or create your `~/.netrc` file.

     Returns `None` if the `~/.netrc` file did not need to be updated and the number of characters written if it did.
    """
    netrc_file = _netrc_path()

    # netrc needs a netrc file; if missing create an empty one.
    if not netrc_file.exists():
        netrc_file.touch()

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


def get_esa_cdse_credentials() -> Tuple[str, str]:
    """Retrieve credentials for ESA's CDSE to download orbits

    This method will prefer to use CDSE credentials from your `~/.netrc` file if they exist,
    otherwise will look for ESA_USERNAME and ESA_PASSWORD environment variables and
    update or create your `~/.netrc` file.

    Returns `username` and `password` .
    """
    _ = _ensure_orbit_credential()
    netrc_file = _netrc_path()
    netrc_credentials = netrc.netrc(netrc_file)
    username, _, password = netrc_credentials.hosts[ESA_CDSE_HOST]
    return username, password

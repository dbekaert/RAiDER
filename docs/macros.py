import requests

import RAiDER


def define_env(env):
    """Macros Hook"""

    @env.macro
    def raider_version():
        return RAiDER.__version__

    @env.macro
    def get_content(url):
        response = requests.get(url)
        response.raise_for_status()
        return response.content.decode()

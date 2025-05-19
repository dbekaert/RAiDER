from pathlib import Path

import s1_orbits


def get_orbits_from_slc_ids(slc_ids: list[str], orbit_directory: str='orbits') -> list[str]:
    orbit_dir = Path(orbit_directory)
    orbit_dir.mkdir(exist_ok=True)

    orbits = {str(s1_orbits.fetch_for_scene(scene, orbit_dir)) for scene in slc_ids}

    return sorted(list(orbits))

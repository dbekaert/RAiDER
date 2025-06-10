from unittest.mock import call, patch

from RAiDER.s1_orbits import get_orbits_from_slc_ids


def test_get_orbits_from_slc_ids(tmp_path):
    with patch('s1_orbits.fetch_for_scene', side_effect=['foo.eof', 'bar.eof', 'foo.eof']) as mock_fetch_for_scene:
        assert get_orbits_from_slc_ids(['scene1', 'scene2', 'scene3'], str(tmp_path)) == ['bar.eof', 'foo.eof']
        mock_fetch_for_scene.assert_has_calls(
            [
                call('scene1', tmp_path),
                call('scene2', tmp_path),
                call('scene3', tmp_path),
            ],
        )

    orbit_dir = tmp_path / 'orbits'
    assert not orbit_dir.exists()
    with patch('s1_orbits.fetch_for_scene', return_value='a.eof') as mock_fetch_for_scene:
        assert get_orbits_from_slc_ids(['scene4'], str(orbit_dir)) == ['a.eof']
        mock_fetch_for_scene.assert_called_once_with('scene4', orbit_dir)
    assert orbit_dir.exists()

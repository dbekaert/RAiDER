'''
When update_rc_file is False, the RC file should NOT be modified if it already
exists.
'''
from pathlib import Path
from platform import system

import pytest

from RAiDER.models import credentials


@pytest.mark.parametrize('model_name', 'ERA5 ERA5T HRES GMAO MERRA2'.split())
def test_updateFalse(tmp_path: Path, model_name):
    # Get the rc file's path
    hidden_ext = '_' if system() == "Windows" else '.'
    rc_filename = credentials.RC_FILENAMES[model_name]
    if rc_filename is None:
        return
    rc_path = tmp_path / (hidden_ext + rc_filename)
    rc_path = rc_path.expanduser()

    # Write some example text to test for
    rc_path.write_text('dummy')

    # Test creation of this model's RC file in current dir
    credentials.check_api(model_name, None, None, tmp_path, update_rc_file=False)

    # Assert the content was unchanged
    content = rc_path.read_text()
    rc_path.unlink()
    assert content == 'dummy', f'{rc_path} was modified'

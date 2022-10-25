def test_raider__main__(script_runner):
    ret = script_runner.run('RAiDER', '-h')
    assert ret.success


def test_raider__main__(script_runner):
    ret = script_runner.run('generateGACOSVRT.py', '-h')
    assert ret.success


def test_raider__main__(script_runner):
    ret = script_runner.run('prepARIA.py', '-h')
    assert ret.success


def test_raider__main__(script_runner):
    ret = script_runner.run('raiderCombine.py', '-h')
    assert ret.success


def test_raider__main__(script_runner):
    ret = script_runner.run('raiderDelay.py', '-h')
    assert ret.success


def test_raider__main__(script_runner):
    ret = script_runner.run('raiderStats.py', '-h')
    assert ret.success


def test_raider__main__(script_runner):
    ret = script_runner.run('raiderDownloadGNSS.py', '-h')
    assert ret.success


def test_raider__main__(script_runner):
    ret = script_runner.run('raiderWeatherModelDebug.py', '-h')
    assert ret.success

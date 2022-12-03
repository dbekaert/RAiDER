
def test_raider__main__2(script_runner):
    ret = script_runner.run('generateGACOSVRT.py')
    assert ret.success


def test_raider__main__3(script_runner):
    ret = script_runner.run('prepARIA.py', '-h')
    assert ret.success


def test_raider__main__4(script_runner):
    ret = script_runner.run('raiderCombine.py', '-h')
    assert ret.success


def test_raider__main__5(script_runner):
    ret = script_runner.run('raider.py', '-h')
    assert ret.success


def test_raider__main__6(script_runner):
    ret = script_runner.run('raiderStats.py', '-h')
    assert ret.success


def test_raider__main__7(script_runner):
    ret = script_runner.run('raiderDownloadGNSS.py', '-h')
    assert ret.success

import io
from collections import namedtuple
from contextlib import redirect_stdout

from pado.__main__ import main

_Output = namedtuple("_Output", "return_code stdout")


def run(func, argv1):
    f = io.StringIO()
    with redirect_stdout(f):
        return_code = func(argv1)
    return _Output(return_code, f.getvalue().rstrip())


def test_no_args():
    assert main([]) == 0


def test_version():
    from pado import __version__
    assert run(main, ['--version']) == (0, __version__)


def test_info_cmd(tmpdir):
    # help
    dataset_path = tmpdir.mkdir("not_a_dataset")
    output = run(main, ["info", str(dataset_path)])
    assert output.return_code == -1
    assert "error" in output.stdout
    assert "not_a_dataset" in output.stdout

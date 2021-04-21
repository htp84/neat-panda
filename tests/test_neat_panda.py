from pathlib import Path
from neat_panda import _get_version_from_toml, __version__


def test_version():
    _path = Path.cwd()
    _file = "pyproject.toml"
    if _path.stem == "tests":
        _path = _path.parent / _file
    else:
        _path = _path / _file
    assert "0.9.8" == __version__ == _get_version_from_toml(_path)

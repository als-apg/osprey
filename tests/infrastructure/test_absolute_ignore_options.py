"""The root conftest makes ``--ignore`` entries absolute before collection.

pytest resolves a relative entry with ``os.getcwd()`` once per entry for every
path it visits, which is the costliest step of collecting the macOS sweep. An
absolute entry skips that call. These tests pin the rewrite and its scope.
"""

from pathlib import Path
from types import SimpleNamespace

from tests.conftest import absolutize_ignore_options


def _config(tmp_path: Path, **options):
    return SimpleNamespace(
        option=SimpleNamespace(**options),
        invocation_params=SimpleNamespace(dir=tmp_path),
    )


def test_relative_entries_resolve_against_the_invocation_directory(tmp_path):
    config = _config(tmp_path, ignore=["tests/cli/test_a.py"], ignore_glob=["tests/*/b_*.py"])
    absolutize_ignore_options(config)
    assert config.option.ignore == [str(tmp_path / "tests/cli/test_a.py")]
    assert config.option.ignore_glob == [str(tmp_path / "tests/*/b_*.py")]


def test_an_absolute_entry_is_kept(tmp_path):
    entry = str(tmp_path / "elsewhere" / "test_c.py")
    config = _config(tmp_path, ignore=[entry], ignore_glob=None)
    absolutize_ignore_options(config)
    assert config.option.ignore == [entry]
    assert config.option.ignore_glob is None


def test_this_session_holds_no_relative_ignore(pytestconfig):
    for name in ("ignore", "ignore_glob"):
        for entry in getattr(pytestconfig.option, name, None) or ():
            assert Path(entry).is_absolute(), (name, entry)

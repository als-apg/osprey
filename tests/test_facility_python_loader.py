"""The neutral by-path loader shared by bluesky plans, health plugins and sequences."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from osprey.utils.module_loading import load_by_path, module_name_for

PREFIX = "_test_facility_python_loader_"


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    yield
    for name in [n for n in sys.modules if n.startswith(PREFIX)]:
        del sys.modules[name]


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def test_module_name_is_prefix_plus_16_hex_of_resolved_path_sha1(tmp_path: Path) -> None:
    path = _write(tmp_path / "mod.py", "X = 1\n")
    expected = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]

    module = load_by_path(path, PREFIX)

    assert module.__name__ == f"{PREFIX}{expected}"
    assert module_name_for(path, PREFIX) == module.__name__
    assert sys.modules[module.__name__] is module
    assert module.X == 1


def test_name_is_stable_per_path_including_relative_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "sub" / "mod.py", "X = 1\n")
    monkeypatch.chdir(tmp_path)

    first = load_by_path(path, PREFIX)
    second = load_by_path(Path("sub") / ".." / "sub" / "mod.py", PREFIX)

    assert first.__name__ == second.__name__
    assert sys.modules[first.__name__] is second


def test_distinct_paths_with_same_stem_get_distinct_names(tmp_path: Path) -> None:
    a = _write(tmp_path / "a" / "orm.py", "WHO = 'a'\n")
    b = _write(tmp_path / "b" / "orm.py", "WHO = 'b'\n")

    mod_a = load_by_path(a, PREFIX)
    mod_b = load_by_path(b, PREFIX)

    assert mod_a.__name__ != mod_b.__name__
    assert (mod_a.WHO, mod_b.WHO) == ("a", "b")
    assert sys.modules[mod_a.__name__] is mod_a


def test_module_is_registered_before_it_executes(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "selfref.py",
        "import sys\nSEEN = __name__ in sys.modules\n",
    )

    assert load_by_path(path, PREFIX).SEEN is True


def test_dataclass_in_loaded_file_resolves_its_module(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dc.py",
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\nclass P:\n    x: int\n",
    )

    assert load_by_path(path, PREFIX).P(1).x == 1


def test_failed_import_leaves_no_sys_modules_entry(tmp_path: Path) -> None:
    path = _write(tmp_path / "boom.py", "raise RuntimeError('boom at import')\n")

    with pytest.raises(RuntimeError, match="boom at import"):
        load_by_path(path, PREFIX)

    assert module_name_for(path, PREFIX) not in sys.modules


def test_base_exception_during_exec_also_pops_and_propagates(tmp_path: Path) -> None:
    path = _write(tmp_path / "interrupt.py", "raise KeyboardInterrupt\n")

    with pytest.raises(KeyboardInterrupt):
        load_by_path(path, PREFIX)

    assert module_name_for(path, PREFIX) not in sys.modules


def test_syntax_error_pops_and_propagates(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.py", "def (:\n")

    with pytest.raises(SyntaxError):
        load_by_path(path, PREFIX)

    assert module_name_for(path, PREFIX) not in sys.modules


def test_failed_reload_does_not_leave_the_previous_good_module_registered(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "flip.py", "X = 1\n")
    load_by_path(path, PREFIX)
    path.write_text("raise ValueError('now broken')\n")

    with pytest.raises(ValueError):
        load_by_path(path, PREFIX)

    assert module_name_for(path, PREFIX) not in sys.modules


def test_missing_file_raises_and_registers_nothing(tmp_path: Path) -> None:
    path = tmp_path / "absent.py"

    with pytest.raises((FileNotFoundError, ImportError)):
        load_by_path(path, PREFIX)

    assert module_name_for(path, PREFIX) not in sys.modules


def test_non_python_suffix_is_refused_with_import_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "data.txt", "X = 1\n")

    with pytest.raises(ImportError, match="no Python import machinery"):
        load_by_path(path, PREFIX)

    assert module_name_for(path, PREFIX) not in sys.modules


def test_prefix_separates_callers_loading_the_same_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "shared.py", "X = 1\n")

    one = load_by_path(path, f"{PREFIX}one_")
    two = load_by_path(path, f"{PREFIX}two_")

    assert one is not two
    assert one.__name__.startswith(f"{PREFIX}one_")
    assert two.__name__.startswith(f"{PREFIX}two_")


def test_bluesky_and_health_loaders_use_the_shared_loader(tmp_path: Path) -> None:
    from osprey.health import plugins as health_plugins
    from osprey.services.bluesky_bridge import plan_loader

    plan = _write(tmp_path / "plan_file.py", "X = 1\n")
    layered = plan_loader._load_plan_module_from_path(plan)
    legacy = plan_loader._load_module_from_path(plan)
    errors: list = []
    plugin = health_plugins._import_plugin_file(str(plan), errors, project_root=None)

    try:
        assert errors == []
        assert layered.__name__ == module_name_for(plan, plan_loader.PLAN_MODULE_PREFIX)
        assert legacy.__name__ == module_name_for(plan, plan_loader.LEGACY_MODULE_PREFIX)
        assert plugin is not None
        assert plugin.__name__ == module_name_for(plan, health_plugins.PLUGIN_MODULE_PREFIX)
    finally:
        for module in (layered, legacy, plugin):
            if module is not None:
                sys.modules.pop(module.__name__, None)

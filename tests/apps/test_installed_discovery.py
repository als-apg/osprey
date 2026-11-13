"""Applications found in installed packages: the ``osprey.apps`` entry points."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from importlib import metadata
from pathlib import Path

import pytest
import yaml

from osprey.apps.errors import AppManifestError
from osprey.apps.loader import load
from osprey.apps.origin import AppOrigin, installed_apps
from tests.apps import _installed
from tests.apps._installed import DIRECTORY, DIST, MODULE, NAME, VERSION, entry_point


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture package on the path; nothing declared unless a test declares it."""
    monkeypatch.syspath_prepend(str(_installed.SITE))
    monkeypatch.delitem(sys.modules, MODULE, raising=False)


pytestmark = pytest.mark.usefixtures("site")


def _rules(err: AppManifestError) -> list[str]:
    return [v.rule for v in err.violations]


def test_an_entry_point_lists_the_app_with_its_directory_and_distribution() -> None:
    found = installed_apps([entry_point()])

    assert found == {
        NAME: AppOrigin(
            kind="installed",
            directory=DIRECTORY,
            distribution=DIST,
            version=VERSION,
            module=MODULE,
        )
    }


def test_discovery_never_imports_the_application() -> None:
    installed_apps([entry_point()])

    assert MODULE not in sys.modules


def test_the_environment_is_read_when_no_entry_points_are_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed.install(monkeypatch)

    assert set(installed_apps()) == {NAME}


def test_two_entry_points_of_one_name_are_refused_naming_both_distributions() -> None:
    with pytest.raises(AppManifestError) as info:
        installed_apps([entry_point(), entry_point(dist="other-dist", version="2.0")])

    assert _rules(info.value) == ["installed_duplicate"]
    assert DIST in info.value.message and "other-dist" in info.value.message


def test_an_entry_point_whose_module_is_no_package_with_an_app_yml_is_refused() -> None:
    with pytest.raises(AppManifestError) as info:
        installed_apps([entry_point(module=f"{MODULE}.tools")])

    assert _rules(info.value) == ["installed_not_a_package"]
    assert DIST in info.value.message


def test_an_entry_point_whose_module_does_not_resolve_is_refused() -> None:
    with pytest.raises(AppManifestError) as info:
        installed_apps([entry_point(module="no_such_app_package")])

    assert _rules(info.value) == ["installed_not_a_package"]


# --- the loader ---------------------------------------------------------------------


def _deployment(tmp_path: Path) -> Path:
    (tmp_path / "apps").mkdir(parents=True)
    return tmp_path


def test_a_named_installed_app_loads_with_its_origin(tmp_path: Path) -> None:
    installed = installed_apps([entry_point()])

    (manifest,) = load(_deployment(tmp_path), [NAME], installed=installed)

    assert manifest.name == NAME
    assert manifest.origin == installed[NAME]
    assert manifest.origin.directory == DIRECTORY


def test_an_unnamed_installed_app_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = installed_apps([entry_point()])
    read: list[Path] = []
    real = Path.read_text

    def spy(self: Path, *args: object, **kwargs: object) -> str:
        read.append(self)
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", spy)
    assert load(_deployment(tmp_path), [], installed=installed) == []
    assert not any(path.is_relative_to(DIRECTORY) for path in read)


def test_deployment_and_installed_apps_load_together_sorted_by_name(tmp_path: Path) -> None:
    root = _deployment(tmp_path)
    (root / "apps" / "aaa").mkdir()
    (root / "apps" / "aaa" / "app.yml").write_text(
        "name: aaa\ncontainers:\n  aaa:\n    runtime: osprey\n", encoding="utf-8"
    )

    manifests = load(root, ["aaa", NAME], installed=installed_apps([entry_point()]))

    assert [m.name for m in manifests] == ["aaa", NAME]
    assert manifests[0].origin is not None
    assert manifests[0].origin.kind == "deployment"
    assert manifests[0].origin.directory == root / "apps" / "aaa"


def _broken_copy(tmp_path: Path, **changes: object) -> Path:
    site = tmp_path / "site"
    shutil.copytree(DIRECTORY, site / "broken_installed_app")
    manifest = site / "broken_installed_app" / "app.yml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data.update(changes)
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    return site


def test_an_installed_manifests_name_must_equal_its_entry_point_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(_broken_copy(tmp_path, name="other-name")))
    installed = installed_apps([entry_point(name="broken", module="broken_installed_app")])

    with pytest.raises(AppManifestError) as info:
        load(_deployment(tmp_path / "root"), ["broken"], installed=installed)

    assert _rules(info.value) == ["name_matches_entry_point"]


def test_a_refusal_in_an_installed_manifest_names_the_installed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(_broken_copy(tmp_path, unknown_key=1)))
    installed = installed_apps([entry_point(name=NAME, module="broken_installed_app")])

    with pytest.raises(AppManifestError) as info:
        load(_deployment(tmp_path / "root"), [NAME], installed=installed)

    where = info.value.violations[0].where
    assert where.startswith(f"broken_installed_app/app.yml (installed by {DIST} {VERSION})")


def test_an_installed_origin_names_its_distribution_and_package() -> None:
    """An installed origin without a package would be reported under a deployment path."""
    with pytest.raises(ValueError, match="installed"):
        AppOrigin(kind="installed", directory=DIRECTORY, distribution=DIST, version=VERSION)


def test_osprey_declares_no_application_of_its_own() -> None:
    """OSPREY's built-in features are framework code; its distribution ships no application."""
    declared = metadata.distribution("osprey-framework").entry_points
    assert [ep.name for ep in declared if ep.group == "osprey.apps"] == []
    assert importlib.util.find_spec("osprey.apps.first_party") is None

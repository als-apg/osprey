"""A real ``osprey build`` stages CANOPY into its render, never into the repo root.

The demo preset's ``pre_build`` step is the text every deployment's own
``profile.yml`` carries (``"{python}" -m osprey.integrations.canopy.stage``).
Run by the build with a payload named and staging on, it writes the payload and
the two wheels into the render being built, so after the swap they sit at
``build/canopy_payload/`` and ``build/*.whl``; every image context gets a copy
at its own root. The sidecar finds the payload from every rendered config.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import _stage_canopy_into_image, build
from osprey.cli.init_cmd import init
from osprey.integrations.canopy import PAYLOAD_DIRNAME
from osprey.integrations.canopy.sidecar import CanopySidecar
from osprey.integrations.canopy.stage import CHECK_ENV, PAYLOAD_ENV, STAGED_FILENAME
from tests.integrations.canopy.test_stage import CANOPY_WHEEL, PYDM_WHEEL, make_payload

STALE_WHEEL = "canopy-0.1.0-py3-none-any.whl"


def _is_canopy(name: str) -> bool:
    folded = name.casefold()
    return folded == PAYLOAD_DIRNAME or (
        folded.endswith(".whl") and folded.startswith(("canopy-", "pydmconverter-"))
    )


def _canopy_entries(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir() if _is_canopy(entry.name))


def _canopy_enabled(config: Path) -> bool:
    panels = yaml.safe_load(config.read_text())["web"]["panels"]
    return bool(panels.get("canopy", {}).get("enabled"))


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("canopy-staging") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "demo", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


def _staged_build(
    materialized: Path, base: Path, *, plant_stale: bool = False
) -> tuple[Path, Path, str]:
    repo = base / "demo"
    shutil.copytree(materialized, repo, symlinks=True)
    if plant_stale:
        (repo / PAYLOAD_DIRNAME).mkdir()
        (repo / PAYLOAD_DIRNAME / "stale.txt").write_text("an older build's copy\n")
        (repo / STALE_WHEEL).write_bytes(b"stale wheel")
    payload = make_payload(base / "payload")
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(CHECK_ENV, raising=False)
        patch.setenv(PAYLOAD_ENV, str(payload))
        result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps"])
    assert result.exit_code == 0, result.output
    return repo, payload, result.output


@pytest.fixture(scope="module")
def built(
    materialized: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Path, Path]]:
    repo, payload, _ = _staged_build(materialized, tmp_path_factory.mktemp("built"))
    yield repo, payload


def _image_contexts(repo: Path) -> list[Path]:
    contexts = sorted(path for path in (repo / "build" / ".image").iterdir() if path.is_dir())
    assert len(contexts) > 1, "the deployment's image and at least one render's"
    return contexts


# --------------------------------------------------------------------------- #
# Where the build puts what it stages
# --------------------------------------------------------------------------- #


def test_the_staging_step_puts_nothing_at_the_deployment_root(built) -> None:
    repo, _ = built
    assert _canopy_entries(repo) == []
    assert not (repo / ".canopy_payload.staging").exists()


def test_the_render_holds_the_payload_and_the_wheels(built) -> None:
    repo, payload = built
    rendered = repo / "build"
    assert _canopy_entries(rendered) == sorted([CANOPY_WHEEL, PAYLOAD_DIRNAME, PYDM_WHEEL])
    assert (rendered / PAYLOAD_DIRNAME / STAGED_FILENAME).read_bytes() == (
        payload / STAGED_FILENAME
    ).read_bytes()
    assert _canopy_enabled(rendered / "config.yml")


def test_every_image_context_holds_the_wheels_and_the_payload(built) -> None:
    repo, payload = built
    for context in _image_contexts(repo):
        assert _canopy_entries(context) == sorted([CANOPY_WHEEL, PAYLOAD_DIRNAME, PYDM_WHEEL])
        assert (context / CANOPY_WHEEL).read_bytes() == (payload / CANOPY_WHEEL).read_bytes()
        assert (context / PAYLOAD_DIRNAME / "viewer" / "index.html").is_file()
        assert not list((context / PAYLOAD_DIRNAME).rglob("*.whl"))


def test_the_image_dockerfile_still_installs_the_wheels_from_its_context_root(built) -> None:
    repo, _ = built
    for context in _image_contexts(repo):
        assert "*.wh[l]" in (context / "build" / "Dockerfile").read_text()


# --------------------------------------------------------------------------- #
# The sidecar finds the payload from every rendered config
# --------------------------------------------------------------------------- #


def _sidecar_payload(config: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    assert config.is_file()
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    return CanopySidecar(config.parent / "agent-data", "", None).payload_dir


def test_the_host_sidecar_finds_the_payload_from_the_deployment_render(
    built, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = built
    found = _sidecar_payload(repo / "build" / "config.yml", monkeypatch)
    assert found == repo / "build" / PAYLOAD_DIRNAME


def test_the_host_sidecar_finds_the_payload_from_a_tier_render(
    built, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = built
    tier_configs = sorted((repo / "build").glob("demo-*/config.yml"))
    assert tier_configs
    for config in tier_configs:
        assert _sidecar_payload(config, monkeypatch) == repo / "build" / PAYLOAD_DIRNAME


def test_the_image_sidecar_finds_the_payload_at_its_context_root(
    built, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = built
    for context in _image_contexts(repo):
        found = _sidecar_payload(context / "build" / "config.yml", monkeypatch)
        assert found == context / PAYLOAD_DIRNAME


# --------------------------------------------------------------------------- #
# Root copies an older build left behind
# --------------------------------------------------------------------------- #


def test_stale_root_copies_never_reach_an_image(
    materialized: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo, _, output = _staged_build(materialized, tmp_path, plant_stale=True)
    for context in _image_contexts(repo):
        assert _canopy_entries(context) == sorted([CANOPY_WHEEL, PAYLOAD_DIRNAME, PYDM_WHEEL])
        assert not (context / PAYLOAD_DIRNAME / "stale.txt").exists()
    assert STALE_WHEEL in output + caplog.text, "the root warning names the leftover"


# --------------------------------------------------------------------------- #
# A panel kept with nothing staged
# --------------------------------------------------------------------------- #


def test_a_build_whose_pre_build_stages_nothing_for_a_kept_panel_fails(
    materialized: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repo = tmp_path / "demo"
    shutil.copytree(materialized, repo, symlinks=True)
    profile = repo / "profile.yml"
    body = profile.read_text()
    step = "    - name: canopy-payload\n      timeout: 30\n      run: '\"{python}\" -m osprey.integrations.canopy.stage'\n"
    assert body.count(step) == 1
    profile.write_text(body.replace(step, "    - name: nothing\n      run: 'true'\n"))
    monkeypatch.delenv(CHECK_ENV, raising=False)
    monkeypatch.setenv(PAYLOAD_ENV, str(make_payload(tmp_path / "payload")))

    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps"])

    assert result.exit_code != 0
    assert "pre_build staged none" in result.output + caplog.text
    assert not (repo / "build" / "config.yml").exists()


# --------------------------------------------------------------------------- #
# The image copy beside the ``osprey up --dev`` framework wheel
# --------------------------------------------------------------------------- #


def test_the_image_copy_leaves_the_dev_framework_wheel_alone(tmp_path: Path) -> None:
    stage_dir = make_payload(tmp_path / "stage" / "src").parent
    source = stage_dir / "src"
    (stage_dir / PAYLOAD_DIRNAME).mkdir()
    (stage_dir / PAYLOAD_DIRNAME / STAGED_FILENAME).write_text("{}")
    for wheel in (CANOPY_WHEEL, PYDM_WHEEL):
        shutil.copy2(source / wheel, stage_dir / wheel)
    image = tmp_path / "image"
    image.mkdir()
    framework = image / "osprey_framework-2026.9.0-py3-none-any.whl"
    framework.write_bytes(b"dev")
    (image / STALE_WHEEL).write_bytes(b"stale")

    _stage_canopy_into_image(stage_dir, image)

    assert framework.read_bytes() == b"dev"
    assert sorted(p.name for p in image.glob("*.whl")) == sorted(
        [CANOPY_WHEEL, PYDM_WHEEL, framework.name]
    )
    assert (image / PAYLOAD_DIRNAME / STAGED_FILENAME).is_file()


def test_the_image_copy_with_nothing_staged_adds_no_canopy(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    image = tmp_path / "image"
    image.mkdir()
    (image / PAYLOAD_DIRNAME).mkdir()
    (image / STALE_WHEEL).write_bytes(b"stale")

    _stage_canopy_into_image(stage_dir, image)

    assert _canopy_entries(image) == []

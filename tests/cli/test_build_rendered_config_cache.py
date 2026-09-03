"""``_rendered_config`` parses a render's ``config.yml`` once per content, never stale."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli import build_cmd
from osprey_connectors import yaml_loader


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    build_cmd._rendered_config_cache.clear()


@pytest.fixture
def parse_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = yaml_loader.safe_load

    def counting(stream):  # type: ignore[no-untyped-def]
        calls.append(stream if isinstance(stream, str) else "<stream>")
        return real(stream)

    monkeypatch.setattr(yaml_loader, "safe_load", counting)
    return calls


def _render(tmp_path: Path, text: str) -> Path:
    render_dir = tmp_path / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "config.yml").write_text(text, encoding="utf-8")
    return render_dir


def test_repeated_reads_parse_once(tmp_path: Path, parse_calls: list[str]) -> None:
    render_dir = _render(tmp_path, "control_system:\n  type: mock\nports: [1, 2]\n")

    first = build_cmd._rendered_config(render_dir)
    second = build_cmd._rendered_config(render_dir)
    third = build_cmd._rendered_config(render_dir)

    assert first == second == third == {"control_system": {"type": "mock"}, "ports": [1, 2]}
    assert len(parse_calls) == 1


def test_callers_get_their_own_copy(tmp_path: Path) -> None:
    render_dir = _render(tmp_path, "control_system:\n  type: mock\n")

    annotated = build_cmd._rendered_config(render_dir)
    annotated["config_dir"] = str(render_dir)
    annotated["control_system"]["type"] = "epics"

    assert build_cmd._rendered_config(render_dir) == {"control_system": {"type": "mock"}}


def test_a_rewrite_is_parsed_afresh(tmp_path: Path, parse_calls: list[str]) -> None:
    render_dir = _render(tmp_path, "container_runtime: auto\n")
    assert build_cmd._rendered_config(render_dir) == {"container_runtime": "auto"}

    # The resolvers rewrite config.yml in place mid-build; same path, new bytes.
    (render_dir / "config.yml").write_text("container_runtime: docker\n", encoding="utf-8")

    assert build_cmd._rendered_config(render_dir) == {"container_runtime": "docker"}
    assert build_cmd._rendered_config(render_dir) == {"container_runtime": "docker"}
    assert len(parse_calls) == 2


def test_an_empty_document_is_an_empty_mapping(tmp_path: Path) -> None:
    render_dir = _render(tmp_path, "")
    assert build_cmd._rendered_config(render_dir) == {}


def test_two_renders_do_not_share_an_entry(tmp_path: Path, parse_calls: list[str]) -> None:
    a = _render(tmp_path / "a", "name: a\n")
    b = _render(tmp_path / "b", "name: b\n")

    assert build_cmd._rendered_config(a) == {"name": "a"}
    assert build_cmd._rendered_config(b) == {"name": "b"}
    assert build_cmd._rendered_config(a) == {"name": "a"}
    assert len(parse_calls) == 2

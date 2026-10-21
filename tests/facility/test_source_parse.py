"""A facility source parses to the same objects, and fails with the same words, on either scanner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.facility import sources
from osprey.facility.sources import load_sources, read_yaml
from osprey_connectors import yaml_loader

_SOURCE = """\
on: off
yes: no
flags: [on, off, yes, no, true, False, ~]
quoted: "on"
hex: 0x10
grouped: 1_000
when: 2026-09-02
far: .inf
"""

_BROKEN = "a: [1, 2\nb: 3\n"


@pytest.fixture(params=["libyaml", "pure-python"])
def scanner(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Each case runs once on the scanner the host has and once with libyaml removed."""
    if request.param == "libyaml":
        if not yaml.__with_libyaml__:
            pytest.skip("libyaml bindings not installed")
    else:
        monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    return request.param


@pytest.mark.usefixtures("scanner")
def test_a_source_parses_as_the_bool_rule_reads_it() -> None:
    parsed = read_yaml(_SOURCE)

    assert parsed == yaml.load(_SOURCE, Loader=sources._Loader)
    assert parsed["on"] == "off"
    assert parsed["yes"] == "no"
    assert parsed["flags"] == ["on", "off", "yes", "no", True, False, None]
    assert parsed["quoted"] == "on"
    assert (parsed["hex"], parsed["grouped"], parsed["far"]) == (16, 1000, float("inf"))
    assert str(parsed["when"]) == "2026-09-02"


@pytest.mark.usefixtures("scanner")
def test_a_source_that_does_not_parse_quotes_the_line(tmp_path: Path) -> None:
    facility = tmp_path / "data" / "facility"
    facility.mkdir(parents=True)
    (facility / "identity.yaml").write_text(_BROKEN, encoding="utf-8")
    with pytest.raises(yaml.YAMLError) as refused:
        yaml.load(_BROKEN, Loader=sources._Loader)
    expected = " ".join(str(refused.value).split())

    (stop,) = load_sources(facility).errors

    assert stop.detail == f"does not parse: {expected}"
    assert "^" in stop.detail


def test_a_source_is_scanned_on_the_scanner_the_host_has(scanner: str) -> None:
    loader = sources._loader_on(yaml_loader.safe_loader())

    assert issubclass(loader, yaml_loader.safe_loader())
    assert loader.yaml_implicit_resolvers is sources._Loader.yaml_implicit_resolvers
    assert (loader is sources._Loader) == (scanner == "pure-python")

"""The provider catalog a deployment repo owns — written, refreshed, stamped.

``providers.yml`` is a source-zone file like ``profile.yml``: ``osprey init``
puts the packaged catalog there, the operator adds gateways to it, and the build
renders the whole file into ``api.providers``. Two rules make that safe to
re-run, and this module is where both are pinned.

* A re-materialization (``osprey init --force``) REFRESHES every entry OSPREY
  ships and KEEPS every entry it does not. Without the refresh a repo never
  receives a corrected endpoint again; without the keep, re-running the command
  costs the operator their own gateways, which nothing else in the repo
  remembers.
* The emitted profile stamps ``provenance.providers_hash`` over the catalog that
  actually landed, so a later build can say the two documents have parted
  without guessing which.

The catalog's own loader, shape and hash are ``tests/profiles/test_providers_catalog.py``'s
subject; what ``osprey init`` does with it is this module's.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.profiles.providers import (
    PROVIDERS_FILENAME,
    compute_providers_hash,
    packaged_catalog_path,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _new(runner: CliRunner, target: Path, *extra: str):
    """Materialize a deployment repo at *target* from the smallest preset."""
    return runner.invoke(init, [str(target), "--preset", "hello-world", "--no-git", *extra])


def _catalog(repo: Path) -> dict:
    """The repo catalog's entries, as the build would read them."""
    return yaml.safe_load((repo / PROVIDERS_FILENAME).read_text(encoding="utf-8"))["providers"]


def _provenance(repo: Path) -> dict:
    """The emitted profile's ``provenance:`` block."""
    return yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))["provenance"]


def _add_gateway(repo: Path, name: str = "house-gateway") -> str:
    """Append an operator's own entry, comment and all. Returns its key variable."""
    variable = f"{name.upper().replace('-', '_')}_API_KEY"
    with (repo / PROVIDERS_FILENAME).open("a", encoding="utf-8") as catalog:
        catalog.write(
            f"  # The gateway in the control room rack.\n"
            f"  {name}:\n"
            f"    api_key: ${{{variable}}}\n"
            f"    base_url: https://gateway.example.invalid/v1\n"
        )
    return variable


# ---------------------------------------------------------------------------
# What a first init writes
# ---------------------------------------------------------------------------


def test_init_writes_the_catalog_beside_the_profile(runner: CliRunner, tmp_path: Path) -> None:
    """The catalog is a repo file from the first minute: the operator edits it
    where they edit the profile, and the build reads it from there."""
    target = tmp_path / "my-facility"

    assert _new(runner, target).exit_code == 0

    written = target / PROVIDERS_FILENAME
    assert written.is_file()
    # Byte-for-byte, so the documentation the packaged catalog carries — what
    # each key means, how to add a gateway — is what the operator opens.
    assert written.read_bytes() == packaged_catalog_path().read_bytes()


def test_the_catalog_belongs_to_the_source_zone(runner: CliRunner, tmp_path: Path) -> None:
    """Everything `--force` may replace is one table, and the catalog is in it —
    that table drives the hold-aside, the rollback and the README's zone row."""
    from osprey.cli.profile_cmd import MATERIALIZED_SOURCE_ENTRIES

    assert PROVIDERS_FILENAME in MATERIALIZED_SOURCE_ENTRIES

    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0
    assert PROVIDERS_FILENAME in (target / "README.md").read_text(encoding="utf-8")


def test_the_emitted_profile_stamps_the_catalog_hash(runner: CliRunner, tmp_path: Path) -> None:
    """`provenance.providers_hash` is the machine-readable record of what the
    catalog held when the profile was written, beside the preset's own hash."""
    target = tmp_path / "my-facility"

    assert _new(runner, target).exit_code == 0

    provenance = _provenance(target)
    assert provenance["providers_hash"] == compute_providers_hash(target / PROVIDERS_FILENAME)
    assert provenance["providers_hash"].startswith("sha256:")


def test_the_profile_comment_sends_the_operator_to_the_catalog() -> None:
    """The `provider:` key documents where a gateway is described. It is
    providers.yml now, and a `config:` spelling of api.providers is refused, so
    a comment still offering that recipe would send the reader into a refusal."""
    from osprey.cli.build_profile_emit import _COMMENTED_TEMPLATES

    template = _COMMENTED_TEMPLATES["provider"]

    assert PROVIDERS_FILENAME in template
    assert "api.providers." not in template


# ---------------------------------------------------------------------------
# Re-materialization: refresh what OSPREY ships, keep what the operator added
# ---------------------------------------------------------------------------


def test_force_preserves_operator_entry(runner: CliRunner, tmp_path: Path) -> None:
    """The acceptance case: an entry OSPREY does not ship survives a re-init.

    Nothing else in the repo remembers the operator's gateway — not the profile,
    which only names one, and not the build zone, which is regenerated. Losing
    it to a command whose whole promise is that re-running is safe would be the
    one unrecoverable thing `--force` could do.
    """
    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0
    _add_gateway(target)

    result = _new(runner, target, "--force")

    assert result.exit_code == 0, result.output
    entries = _catalog(target)
    assert entries["house-gateway"]["base_url"] == "https://gateway.example.invalid/v1"
    assert entries["house-gateway"]["api_key"] == "${HOUSE_GATEWAY_API_KEY}"
    # Its comment is part of the entry: the catalog is documentation as much as
    # data, and a merge that kept only the values would be a lossy edit of a
    # file the operator owns.
    assert "The gateway in the control room rack." in (target / PROVIDERS_FILENAME).read_text(
        encoding="utf-8"
    )


def test_force_preserves_an_entry_under_a_commented_providers_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A trailing comment on the `providers:` line is ordinary YAML, and the
    slicer has to read past it.

    Matching only the bare key found no entries at all in such a file, so every
    one of the operator's gateways read as lost and the merge refused — a
    spurious refusal on a file that is perfectly valid.
    """
    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0
    catalog_file = target / PROVIDERS_FILENAME
    catalog_file.write_text(
        catalog_file.read_text(encoding="utf-8").replace(
            "\nproviders:\n", "\nproviders:  # every gateway we can reach\n"
        ),
        encoding="utf-8",
    )
    _add_gateway(target)

    result = _new(runner, target, "--force")

    assert result.exit_code == 0, result.output
    assert "house-gateway" in _catalog(target)


def test_force_refreshes_an_entry_osprey_ships(runner: CliRunner, tmp_path: Path) -> None:
    """The other half of the rule: a packaged entry goes back to the packaged
    value, which is how a corrected endpoint reaches a repo at all."""
    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0

    catalog_file = target / PROVIDERS_FILENAME
    catalog_file.write_text(
        catalog_file.read_text(encoding="utf-8").replace(
            "https://api.cborg.lbl.gov/v1", "https://stale.example.invalid/v1"
        ),
        encoding="utf-8",
    )
    _add_gateway(target)

    assert _new(runner, target, "--force").exit_code == 0

    entries = _catalog(target)
    assert entries["cborg"]["base_url"] == "https://api.cborg.lbl.gov/v1"
    assert "house-gateway" in entries


def test_force_restamps_the_hash_over_the_merged_catalog(runner: CliRunner, tmp_path: Path) -> None:
    """The stamp describes the file that ended up in the repo, not the packaged
    copy — otherwise every repo with a gateway of its own reads as drifted the
    moment it is re-initialized."""
    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0
    _add_gateway(target)

    assert _new(runner, target, "--force").exit_code == 0

    stamped = _provenance(target)["providers_hash"]
    assert stamped == compute_providers_hash(target / PROVIDERS_FILENAME)
    assert stamped != compute_providers_hash(packaged_catalog_path())


def test_a_catalog_that_does_not_parse_refuses_the_re_init(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Refusing is the only honest answer: the merge cannot tell an operator's
    entry from a packaged one in a file it cannot read, and the repo it would
    overwrite is the only copy of those entries."""
    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0
    (target / PROVIDERS_FILENAME).write_text("providers: [not, a, mapping]\n", encoding="utf-8")

    result = _new(runner, target, "--force")

    assert result.exit_code != 0
    assert PROVIDERS_FILENAME in result.output
    # Refused before anything moved: the operator's file is still there to fix.
    assert (target / PROVIDERS_FILENAME).read_text(encoding="utf-8").startswith("providers:")
    assert (target / "profile.yml").is_file()


# ---------------------------------------------------------------------------
# `.env` seeding reads the catalog
# ---------------------------------------------------------------------------


def test_the_key_variable_comes_from_the_entry(runner: CliRunner, tmp_path: Path) -> None:
    """A gateway OSPREY does not ship has no entry in any code-level registry,
    so its `api_key:` is the only place its variable is named."""
    from osprey.cli.profile_cmd import _catalog_key_variables

    variables = _catalog_key_variables(
        {
            "house-gateway": {"api_key": "${HOUSE_GATEWAY_API_KEY}"},
            "with-default": {"api_key": "${WITH_DEFAULT_KEY:-}"},
            "local": {"api_key": "EMPTY"},
            "keyless": {"base_url": "http://127.0.0.1:8000/v1"},
        }
    )

    assert variables == {
        "house-gateway": "HOUSE_GATEWAY_API_KEY",
        "with-default": "WITH_DEFAULT_KEY",
    }


def test_the_packaged_catalog_agrees_with_the_registry() -> None:
    """The catalog is laid over the registry, so a stock repo's seeding is
    unchanged only while the two name the same variable for every provider."""
    from osprey.cli.profile_cmd import _catalog_key_variables
    from osprey.cli.templates.scaffolding import provider_api_key_entries
    from osprey.profiles.providers import load_provider_catalog

    from_catalog = _catalog_key_variables(load_provider_catalog(None).entries)
    for entry in provider_api_key_entries():
        if entry["provider"] in from_catalog:
            assert from_catalog[entry["provider"]] == entry["var"]


# ---------------------------------------------------------------------------
# Catalog drift is a note, never a refusal
# ---------------------------------------------------------------------------


def _drift(repo: Path):
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.cli.build_profile_drift import preset_drift_report

    profile_path = (repo / "profile.yml").resolve()
    resolved, _ = resolve_build_profile(profile_path, None)
    assert resolved.provenance is not None
    return preset_drift_report(profile_path, resolved.provenance)


def test_a_stock_catalog_draws_no_note(runner: CliRunner, tmp_path: Path) -> None:
    """A repo whose catalog is the one OSPREY ships has not parted from it."""
    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0

    assert _drift(target).note is None


def test_a_customized_catalog_is_a_note_and_not_a_finding(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Adding a gateway is the feature working, so it is never a refusal. What
    is worth saying is that the packaged entries stop arriving on their own, and
    which verb brings them back."""
    target = tmp_path / "my-facility"
    assert _new(runner, target).exit_code == 0
    _add_gateway(target)

    report = _drift(target)

    assert report.note is not None
    assert PROVIDERS_FILENAME in report.note
    assert "osprey profile expand --providers" in report.note
    assert report.unmarked == []

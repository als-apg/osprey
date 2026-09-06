"""``api.providers`` comes from ``providers.yml``, and from nowhere else.

Three facts, one source. The build renders the catalog beside ``profile.yml``
into the render context, so the framework template writes ``api.providers``
from a file the operator holds rather than from a block baked into a template.
The build says which catalog it read, because the repo's own file and the
packaged fallback answer to different edits. And the two ways a profile could
say the same thing a second time are refused: a ``config:`` entry under
``api.providers`` (a second catalog the render silently wins over) and a
``provider:`` naming an entry the catalog does not declare (a name with no
endpoint behind it).

The refusals are what a build asks BEFORE it renders, so those cases cost a
profile parse rather than a render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_model import provider_catalog_key_errors
from osprey.cli.templates.manager import TemplateManager
from osprey.profiles.providers import PROVIDERS_FILENAME, load_provider_catalog

#: A catalog an operator wrote: the built-in name the profiles below select,
#: plus a gateway no packaged catalog knows. Both are entries, so both are
#: names `provider:` may take and both must reach the render context.
_REPO_CATALOG = {
    "providers": {
        "house": {
            "api_key": "${HOUSE_API_KEY}",
            "base_url": "https://gateway.example.org/v1",
            "models": {"haiku": "h-small", "sonnet": "h-mid", "opus": "h-large"},
        },
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}",
            "base_url": "https://api.anthropic.com",
            "models": {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5"},
        },
    }
}


def _repo(tmp_path: Path, name: str, profile: dict[str, Any], catalog: Any = None) -> Path:
    """A deployment repo holding *profile*, and *catalog* when one is given."""
    repo = tmp_path / name
    repo.mkdir()
    (repo / "data").mkdir()
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    if catalog is not None:
        (repo / PROVIDERS_FILENAME).write_text(
            yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8"
        )
    return repo


def _hello_world(**overrides: Any) -> dict[str, Any]:
    """A profile that builds: the smallest bundle, with a named provider."""
    return {
        "name": "Demo Facility",
        "extends": "hello-world",
        "data": "data",
        "provider": "anthropic",
        **overrides,
    }


def _run(repo: Path) -> Result:
    """``osprey build`` over *repo*, with the venv and lifecycle skipped."""
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _build_capturing_contexts(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Result, list[dict[str, Any]]]:
    """Build *repo*, returning the result and every render context it built.

    Spies on the real ``create_project`` rather than replacing it, so the build
    still renders: the assertion is about what the template is HANDED, and a
    context that never reached a render would prove nothing about the render.
    """
    contexts: list[dict[str, Any]] = []
    original = TemplateManager.create_project

    def spy(self: TemplateManager, *args: Any, **kwargs: Any) -> Any:
        contexts.append(kwargs["context"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TemplateManager, "create_project", spy)
    return _run(repo), contexts


class TestRenderContext:
    """What the framework template is handed to render ``api.providers`` from."""

    def test_packaged_catalog_reaches_every_render(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repo with no catalog of its own renders the packaged entries."""
        result, contexts = _build_capturing_contexts(
            _repo(tmp_path, "packaged", _hello_world()), monkeypatch
        )

        assert result.exit_code == 0, result.output
        assert contexts, "the build rendered no project"
        packaged = load_provider_catalog(None).entries
        for context in contexts:
            assert context["provider_catalog"] == packaged

    def test_repo_catalog_replaces_the_packaged_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``providers.yml`` beside the profile is the whole catalog.

        The packaged entries the repo file does not name are gone from the
        context, which is the point of the file: an operator removing a
        decommissioned gateway removes it from the render.
        """
        repo = _repo(tmp_path, "own", _hello_world(), _REPO_CATALOG)

        result, contexts = _build_capturing_contexts(repo, monkeypatch)

        assert result.exit_code == 0, result.output
        for context in contexts:
            assert context["provider_catalog"] == _REPO_CATALOG["providers"]
            assert set(context["provider_catalog"]) == {"house", "anthropic"}


class TestBuildSummary:
    """The build says which catalog it read."""

    def test_names_the_repo_catalog(self, tmp_path: Path) -> None:
        result = _run(_repo(tmp_path, "own", _hello_world(), _REPO_CATALOG))

        assert result.exit_code == 0, result.output
        assert f"provider catalog {PROVIDERS_FILENAME}" in result.output

    def test_names_the_packaged_catalog(self, tmp_path: Path) -> None:
        """Named, not silent: the repo has no file to edit, and an operator
        adding a gateway has to be told that."""
        result = _run(_repo(tmp_path, "packaged", _hello_world()))

        assert result.exit_code == 0, result.output
        assert "provider catalog packaged" in result.output
        assert f"provider catalog {PROVIDERS_FILENAME}\n" not in result.output


class TestProviderMustBeInTheCatalog:
    """``provider:`` may name a catalog entry and nothing else."""

    def test_unknown_provider_is_refused_naming_the_packaged_catalog(self, tmp_path: Path) -> None:
        result = _run(_repo(tmp_path, "unknown", _hello_world(provider="no-such-gateway")))

        assert result.exit_code != 0
        assert "no-such-gateway" in result.output
        assert f"the packaged {PROVIDERS_FILENAME}" in result.output

    def test_unknown_provider_lists_the_catalog_entries_sorted(self, tmp_path: Path) -> None:
        """The fix is a name from the file, so the refusal hands over the names."""
        result = _run(_repo(tmp_path, "listing", _hello_world(provider="nope"), _REPO_CATALOG))

        assert result.exit_code != 0
        assert "anthropic, house" in " ".join(result.output.split())

    def test_a_repo_catalog_narrows_what_provider_may_name(self, tmp_path: Path) -> None:
        """Replacement, not merge: a packaged name the repo file drops is gone.

        This is the case a merge would get wrong — ``anthropic`` is in the
        packaged catalog, and the profile still may not name it once the repo
        declares a catalog without it.
        """
        narrow = {"providers": {"house": _REPO_CATALOG["providers"]["house"]}}

        result = _run(_repo(tmp_path, "narrow", _hello_world(), narrow))

        assert result.exit_code != 0
        assert "anthropic" in result.output
        assert PROVIDERS_FILENAME in result.output

    def test_a_missing_provider_still_names_the_catalog(self, tmp_path: Path) -> None:
        """The empty case names the same file as the wrong-name case, so both
        refusals send an author to one place."""
        # Stated as null rather than dropped: the preset this profile extends
        # names a provider, so a profile that merely omits the key inherits one.
        profile = _hello_world(provider=None)

        result = _run(_repo(tmp_path, "none", profile, _REPO_CATALOG))

        assert result.exit_code != 0
        assert "names no provider" in result.output
        assert PROVIDERS_FILENAME in result.output
        assert "anthropic, house" in " ".join(result.output.split())


class TestConfigMayNotDeclareAProvider:
    """``config: api.providers.*`` is a second catalog, and it is refused."""

    @pytest.mark.parametrize(
        "spelling",
        [
            pytest.param(
                {"api.providers.house.base_url": "https://gateway.example.org/v1"},
                id="whole-dotted-key",
            ),
            pytest.param(
                {"api.providers": {"house": {"base_url": "https://gateway.example.org/v1"}}},
                id="dotted-prefix-over-a-mapping",
            ),
            pytest.param(
                {"api": {"providers": {"house": {"base_url": "https://x.example.org/v1"}}}},
                id="fully-nested",
            ),
            pytest.param(
                {"api": {"providers.house.base_url": "https://x.example.org/v1"}},
                id="mixed-nesting",
            ),
        ],
    )
    def test_every_spelling_is_refused(self, spelling: dict[str, Any]) -> None:
        """All four reach the same rendered leaf, so all four are one refusal."""
        errors = provider_catalog_key_errors(spelling)

        assert errors, f"{spelling} was accepted"
        assert all(PROVIDERS_FILENAME in error for error in errors)
        assert all("api.providers.house" in error for error in errors)

    def test_the_bare_branch_is_refused_by_name(self) -> None:
        """An empty ``api.providers:`` names no provider to list, so the
        refusal names the branch instead of nothing."""
        errors = provider_catalog_key_errors({"api.providers": {}})

        assert errors == [
            f"config: api.providers is rendered by the build from {PROVIDERS_FILENAME}. "
            f"Declare the provider there and remove it from profile.yml."
        ]

    def test_each_declared_provider_is_named(self) -> None:
        """One error per entry: an author moving them needs each name."""
        errors = provider_catalog_key_errors(
            {
                "api.providers.house.base_url": "https://a.example.org/v1",
                "api.providers.annex.base_url": "https://b.example.org/v1",
            }
        )

        assert len(errors) == 2
        assert any("api.providers.annex" in error for error in errors)
        assert any("api.providers.house" in error for error in errors)

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="empty"),
            pytest.param({"api.timeout_seconds": 30}, id="another-api-key"),
            pytest.param({"api": {"timeout_seconds": 30}}, id="another-api-key-nested"),
            pytest.param({"web.theme": "dark"}, id="unrelated-branch"),
            pytest.param("not a mapping", id="non-mapping"),
            pytest.param(None, id="none"),
        ],
    )
    def test_a_block_declaring_no_provider_passes(self, config: Any) -> None:
        assert provider_catalog_key_errors(config) == []

    def test_the_build_refuses_it_before_rendering(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Wired into ``BuildProfile.validate``, not only callable on its own.

        Profile validation is where the build reports it, so the refusal
        arrives on the build logger rather than as click's own usage error.
        """
        repo = _repo(
            tmp_path,
            "declared",
            _hello_world(config={"api.providers.house.base_url": "https://x.example.org/v1"}),
        )

        with caplog.at_level("ERROR"):
            result = _run(repo)

        assert result.exit_code != 0
        assert "api.providers.house" in caplog.text
        assert PROVIDERS_FILENAME in caplog.text

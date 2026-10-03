"""``provider:`` names the agent's chat provider, never an embeddings-only one.

The packaged catalog carries ``llama-cpp``, the site's llama-server, which
serves image and text embeddings and no chat. The build refuses it as the
profile's ``provider:`` with a sentence that says where it belongs instead, and
the lists of names the build's refusals hand over leave it out — while a
catalog-only gateway with no adapter class (``house``) stays listed, because
only the registry's chat fact decides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.build_cmd import build
from osprey.profiles.providers import PROVIDERS_FILENAME, load_provider_catalog

#: The refusal the build and the agent resolver both give for ``llama-cpp``.
REFUSAL = (
    "llama-cpp serves embeddings only, no chat; name it as an embedding module's "
    "provider (ariel.enhancement_modules.image_embedding.provider or "
    "text_embedding.provider)"
)


def _catalog() -> dict[str, Any]:
    """A repo catalog: a gateway with no adapter class, and the llama-server."""
    return {
        "providers": {
            "house": {
                "api_key": "${HOUSE_API_KEY}",
                "base_url": "https://gateway.example.org/v1",
                "default_model": "h-mid",
                "models": ["h-small", "h-mid"],
            },
            "llama-cpp": dict(load_provider_catalog(None).entries["llama-cpp"]),
        }
    }


def _repo(tmp_path: Path, provider: str | None, catalog: dict[str, Any] | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data").mkdir()
    profile = {
        "name": "Demo Facility",
        "extends": "hello-world",
        "data": "data",
        "provider": provider,
    }
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    if catalog is not None:
        (repo / PROVIDERS_FILENAME).write_text(
            yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8"
        )
    return repo


def _run(repo: Path) -> Result:
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _flat(result: Result) -> str:
    """The output with line wrapping undone, so a sentence reads whole."""
    return " ".join(result.output.split())


def test_the_packaged_llama_cpp_entry_is_refused_as_the_provider(tmp_path: Path) -> None:
    result = _run(_repo(tmp_path, "llama-cpp"))

    assert result.exit_code != 0
    assert REFUSAL in _flat(result)


def test_a_repo_catalogs_llama_cpp_entry_is_refused_as_the_provider(tmp_path: Path) -> None:
    result = _run(_repo(tmp_path, "llama-cpp", _catalog()))

    assert result.exit_code != 0
    assert REFUSAL in _flat(result)
    assert "Name one of house as `provider:`" in _flat(result)


@pytest.mark.parametrize("provider", ["nope", None], ids=["unknown-name", "no-provider"])
def test_the_refusal_lists_the_catalog_only_gateway_and_not_llama_cpp(
    tmp_path: Path, provider: str | None
) -> None:
    result = _run(_repo(tmp_path, provider, _catalog()))

    assert result.exit_code != 0
    flat = _flat(result)
    assert "one of house" in flat
    assert "llama-cpp" not in flat


def test_the_packaged_names_leave_llama_cpp_out(tmp_path: Path) -> None:
    result = _run(_repo(tmp_path, "no-such-gateway"))

    assert result.exit_code != 0
    flat = _flat(result)
    assert "anthropic" in flat
    assert "llama-cpp" not in flat

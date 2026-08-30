"""Seeding Claude Code's first-run state for containerized deployments.

A fresh container volume is, from Claude Code's point of view, a brand-new
machine: the interactive CLI walks the operator through onboarding (theme
picker, security notes, terminal setup), asks whether to trust the project
folder, and — under a raw-key provider — whether to use the ambient API key.
None of that is an operator's decision in a deployment OSPREY rendered: the
render already states the theme, the permissions, and the provider. Worse, the
trust dialog is not cosmetic: the rendered ``permissions.allow`` list does not
apply until the folder is trusted, so an unseeded first session runs with a
degraded permission surface.

:func:`osprey.deployment.claude_state_seed.seed_claude_state` writes the state
Claude Code would have recorded had the operator answered, into the
``.claude.json`` the container's ``CLAUDE_CONFIG_DIR``/``HOME`` names. The
properties asserted here:

- **Merge-only.** A key that exists is never rewritten — a returning
  operator's live volume keeps every choice they made, including an explicit
  ``hasCompletedOnboarding: false``.
- **Never clobber.** A file that does not parse is left byte-for-byte alone;
  losing an operator's OAuth state to a seed step would be strictly worse
  than showing the prompts.
- **Deterministic trust key.** Trust is recorded against the resolved render
  directory — the cwd every launcher starts Claude Code in — exactly as
  Claude Code keys it for a non-git directory.
- **Provider-conditional key approval.** Only a provider whose auth reaches
  Claude Code as ``ANTHROPIC_API_KEY`` triggers the CLI's key-approval
  prompt, so only that shape is seeded; token-auth proxies get nothing.
- **Idempotent.** A second run against its own output writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.deployment.claude_state_seed import CLAUDE_STATE_FILENAME, seed_claude_state

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def render_dir(tmp_path: Path) -> Path:
    """A minimal render: a directory holding a config.yml with a pinned CLI."""
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(
        "claude_code:\n  provider: anthropic\n  cli_version: '2.1.239'\n"
    )
    return render


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """The volume-backed directory ``CLAUDE_CONFIG_DIR`` points at."""
    d = tmp_path / "claude-config"
    d.mkdir()
    return d


def _seed(render: Path, config_dir: Path, **env: str) -> list[str]:
    return seed_claude_state(render, env={"CLAUDE_CONFIG_DIR": str(config_dir), **env})


def _state(config_dir: Path) -> dict:
    return json.loads((config_dir / CLAUDE_STATE_FILENAME).read_text())


# ── fresh volume ─────────────────────────────────────────────────────────────


def test_fresh_volume_gets_onboarding_trust_and_version(render_dir, config_dir):
    seeded = _seed(render_dir, config_dir)

    state = _state(config_dir)
    assert state["hasCompletedOnboarding"] is True
    assert state["lastOnboardingVersion"] == "2.1.239"
    assert state["projects"][str(render_dir)]["hasTrustDialogAccepted"] is True
    assert seeded  # every action is reported for the entrypoint log


def test_config_dir_is_created_when_missing(render_dir, tmp_path):
    """A named volume mounts as an existing dir, but a bare HOME may not hold one."""
    config_dir = tmp_path / "not-yet"
    seed_claude_state(render_dir, env={"CLAUDE_CONFIG_DIR": str(config_dir)})
    assert (config_dir / CLAUDE_STATE_FILENAME).is_file()


def test_home_fallback_when_config_dir_unset(render_dir, tmp_path):
    """The single-user image sets HOME only; state lands at ~/.claude.json."""
    home = tmp_path / "home"
    home.mkdir()
    seed_claude_state(render_dir, env={"HOME": str(home)})
    assert (home / CLAUDE_STATE_FILENAME).is_file()


def test_no_target_dir_is_a_reported_noop(render_dir):
    """Neither CLAUDE_CONFIG_DIR nor HOME: nowhere to write, and no crash."""
    assert seed_claude_state(render_dir, env={}) == []


# ── merge-only semantics ─────────────────────────────────────────────────────


def test_existing_keys_and_unrelated_state_survive(render_dir, config_dir):
    """The seed adds what is missing and rewrites nothing that exists."""
    (config_dir / CLAUDE_STATE_FILENAME).write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": False,  # an explicit choice, kept
                "oauthAccount": {"email": "op@example.org"},  # untouched
                "projects": {
                    "/somewhere/else": {"hasTrustDialogAccepted": False},
                },
            }
        )
    )

    _seed(render_dir, config_dir)

    state = _state(config_dir)
    assert state["hasCompletedOnboarding"] is False
    assert state["oauthAccount"] == {"email": "op@example.org"}
    assert state["projects"]["/somewhere/else"] == {"hasTrustDialogAccepted": False}
    # ...while the render's own trust entry is still added beside it.
    assert state["projects"][str(render_dir)]["hasTrustDialogAccepted"] is True


def test_existing_project_entry_gains_only_the_missing_key(render_dir, config_dir):
    (config_dir / CLAUDE_STATE_FILENAME).write_text(
        json.dumps({"projects": {str(render_dir): {"exampleFiles": ["a.py"]}}})
    )

    _seed(render_dir, config_dir)

    entry = _state(config_dir)["projects"][str(render_dir)]
    assert entry["exampleFiles"] == ["a.py"]
    assert entry["hasTrustDialogAccepted"] is True


def test_corrupt_state_file_is_left_untouched(render_dir, config_dir):
    """A parse failure must not cost the operator their file."""
    corrupt = '{"oauthAccount": '  # truncated write
    (config_dir / CLAUDE_STATE_FILENAME).write_text(corrupt)

    assert _seed(render_dir, config_dir) == []
    assert (config_dir / CLAUDE_STATE_FILENAME).read_text() == corrupt


def test_second_run_is_a_silent_noop(render_dir, config_dir):
    _seed(render_dir, config_dir)
    before = (config_dir / CLAUDE_STATE_FILENAME).read_text()

    assert _seed(render_dir, config_dir) == []
    assert (config_dir / CLAUDE_STATE_FILENAME).read_text() == before


# ── provider-conditional API-key approval ────────────────────────────────────


def test_anthropic_key_is_pre_approved_by_its_last_20_chars(render_dir, config_dir):
    key = "sk-ant-api03-" + "x" * 40
    _seed(render_dir, config_dir, ANTHROPIC_API_KEY=key)

    approved = _state(config_dir)["customApiKeyResponses"]["approved"]
    assert approved == [key[-20:]]


def test_token_auth_provider_seeds_no_key_approval(config_dir, tmp_path):
    """A proxy provider authenticates with ANTHROPIC_AUTH_TOKEN — no prompt exists."""
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(
        "claude_code:\n"
        "  provider: cborg\n"
        "api:\n"
        "  providers:\n"
        "    cborg:\n"
        "      models: {haiku: anthropic/claude-haiku}\n"
    )

    _seed(render, config_dir, CBORG_API_KEY="cborg-secret", ANTHROPIC_AUTH_TOKEN="cborg-secret")

    assert "customApiKeyResponses" not in _state(config_dir)


def test_missing_key_value_seeds_no_approval(render_dir, config_dir):
    """Provider says ANTHROPIC_API_KEY but the container got no key: nothing to approve."""
    _seed(render_dir, config_dir)
    assert "customApiKeyResponses" not in _state(config_dir)


def test_already_rejected_digest_is_respected(render_dir, config_dir):
    key = "sk-ant-api03-" + "y" * 40
    (config_dir / CLAUDE_STATE_FILENAME).write_text(
        json.dumps({"customApiKeyResponses": {"approved": [], "rejected": [key[-20:]]}})
    )

    _seed(render_dir, config_dir, ANTHROPIC_API_KEY=key)

    responses = _state(config_dir)["customApiKeyResponses"]
    assert responses["approved"] == []
    assert responses["rejected"] == [key[-20:]]


# ── degraded configs ─────────────────────────────────────────────────────────


def test_render_without_config_still_seeds_onboarding_and_trust(config_dir, tmp_path):
    """No config.yml: no version, no provider — the universal keys still land."""
    render = tmp_path / "bare"
    render.mkdir()

    _seed(render, config_dir)

    state = _state(config_dir)
    assert state["hasCompletedOnboarding"] is True
    assert "lastOnboardingVersion" not in state
    assert state["projects"][str(render)]["hasTrustDialogAccepted"] is True

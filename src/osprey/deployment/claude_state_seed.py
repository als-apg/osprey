"""Seed Claude Code's first-run state so a container's first session starts clean.

Claude Code keeps per-machine application state in ``.claude.json`` — under
``$CLAUDE_CONFIG_DIR`` when that is set, beside ``$HOME`` otherwise. A fresh
container volume holds neither, so the first interactive session walks the
operator through the CLI's own onboarding: a theme picker, security notes, a
terminal-setup offer, then the workspace-trust dialog, and — when the provider
hands Claude Code a raw ``ANTHROPIC_API_KEY`` — an approval prompt for that
key. For a developer on their own machine those are real questions; for an
operator dropped into a deployment OSPREY rendered they are noise at best,
and the trust dialog is a genuine defect: Claude Code applies a project's
``permissions.allow`` rules only after the folder is trusted, so the render's
carefully-built allow list is inert until the operator clicks through a dialog
they have no context to evaluate.

This module writes the state Claude Code would have recorded had the operator
answered, once, from the container entrypoint's root phase — the same phase
that regenerates drifted artifacts and restores scaffold bodies, and for the
same reason: it is a write into state the serving process must find already
in place. Three of the four keys are documented ground
(``projects[<path>].hasTrustDialogAccepted`` is the documented manual-trust
mechanism; ``theme`` never needs seeding because skipping onboarding lands on
Claude Code's default); ``customApiKeyResponses`` is observed app state, so
that one seed is best-effort by design.

The contract, asserted in ``tests/deployment/test_claude_state_seed.py``:

- **Merge-only**: a key that exists is never rewritten, so a returning
  operator's volume keeps every choice they made.
- **Never clobber**: a file that does not parse is left alone — losing an
  operator's OAuth state would be strictly worse than showing the prompts.
- **Fail open**: like the entrypoint's other maintenance steps, any failure
  is reported and the container still starts, prompts and all.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from osprey.utils.logger import get_logger

logger = get_logger("deployment.claude_state_seed")

#: Claude Code's per-machine state file, relative to the config dir / HOME.
CLAUDE_STATE_FILENAME = ".claude.json"

#: The one auth shape whose interactive session prompts for key approval.
#: Token-auth providers (``ANTHROPIC_AUTH_TOKEN``) never see the prompt.
_PROMPTED_AUTH_ENV_VAR = "ANTHROPIC_API_KEY"

#: How Claude Code digests a key into ``customApiKeyResponses``: the trimmed
#: key's last 20 characters. Observed in the pinned CLI, not documented — the
#: reason this particular seed stays best-effort.
_API_KEY_DIGEST_CHARS = 20


def seed_claude_state(
    render_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
    owner_user: str | None = None,
) -> list[str]:
    """Write the missing first-run keys into Claude Code's state file.

    Args:
        render_dir: The rendered project directory — the cwd every launcher
            starts Claude Code in, and therefore the exact path Claude Code
            keys workspace trust on (the render ships no ``.git``, so trust is
            keyed on the directory itself, not a repository root).
        env: The environment to resolve ``CLAUDE_CONFIG_DIR``/``HOME`` and the
            provider's key from. Defaults to ``os.environ``; injectable for
            tests.
        owner_user: When set and the process runs as root, ``chown`` the state
            file (and the config dir, if this call created it) to this user —
            the entrypoint's privilege-drop target. Root's file in a volume
            the dropped process must rewrite would be worse than no seed.

    Returns:
        Human-readable descriptions of the keys seeded, for the entrypoint
        log. Empty when everything was already in place — or when nothing
        could be done, each case having logged why.
    """
    if env is None:
        env = os.environ

    base = env.get("CLAUDE_CONFIG_DIR", "").strip() or env.get("HOME", "").strip()
    if not base:
        logger.warning(
            "Neither CLAUDE_CONFIG_DIR nor HOME is set; cannot locate Claude Code's "
            "state file, so the first session will show the interactive setup prompts."
        )
        return []

    render_dir = Path(render_dir).resolve()
    base_dir = Path(base)
    target = base_dir / CLAUDE_STATE_FILENAME

    state, ok = _load_state(target)
    if not ok:
        return []

    seeded: list[str] = []

    if "hasCompletedOnboarding" not in state:
        state["hasCompletedOnboarding"] = True
        seeded.append("onboarding marked complete")

    cli_version = _pinned_cli_version(render_dir)
    if cli_version and "lastOnboardingVersion" not in state:
        state["lastOnboardingVersion"] = cli_version
        seeded.append(f"onboarding version {cli_version}")

    projects = state.setdefault("projects", {})
    entry = projects.setdefault(str(render_dir), {})
    if "hasTrustDialogAccepted" not in entry:
        entry["hasTrustDialogAccepted"] = True
        seeded.append(f"workspace trust for {render_dir}")

    digest = _api_key_digest(render_dir, env)
    if digest is not None:
        responses = state.setdefault("customApiKeyResponses", {})
        approved = responses.setdefault("approved", [])
        rejected = responses.setdefault("rejected", [])
        if digest not in approved and digest not in rejected:
            approved.append(digest)
            seeded.append("provider API key pre-approved")

    if not seeded:
        return []

    base_dir.mkdir(parents=True, exist_ok=True)

    # Atomic replace: Claude Code itself rewrites this file through a tmp +
    # rename, and a torn write here would trip its corrupt-config recovery on
    # the very first launch this seed exists to smooth.
    tmp = target.with_name(target.name + ".osprey-seed")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(tmp, target)

    _hand_ownership(target, base_dir, owner_user)
    return seeded


def _load_state(target: Path) -> tuple[dict, bool]:
    """Read the existing state file. ``(state, ok)``; ``ok=False`` means abort.

    Missing file → empty state, proceed. Unparseable or non-object content →
    abort without touching the file: this seed must never be the reason an
    operator's OAuth session or MCP registrations are lost.
    """
    if not target.is_file():
        return {}, True
    try:
        state = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Existing %s could not be parsed (%s); leaving it untouched — the "
            "first session may show the interactive setup prompts.",
            target,
            exc,
        )
        return {}, False
    if not isinstance(state, dict):
        logger.warning("Existing %s is not a JSON object; leaving it untouched.", target)
        return {}, False
    return state, True


def _pinned_cli_version(render_dir: Path) -> str | None:
    """The render's ``claude_code.cli_version`` pin, if it states one."""
    config_path = render_dir / "config.yml"
    if not config_path.is_file():
        return None
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text()) or {}
        version = (config.get("claude_code") or {}).get("cli_version")
        return version.strip() if isinstance(version, str) and version.strip() else None
    except Exception as exc:  # noqa: BLE001 — a version is nice-to-have, never blocking
        logger.warning("Could not read cli_version from %s (%r)", config_path, exc)
        return None


def _api_key_digest(render_dir: Path, env: Mapping[str, str]) -> str | None:
    """The approval digest for the provider's key, when the prompt would fire.

    Only a resolved provider whose auth reaches Claude Code as
    ``ANTHROPIC_API_KEY`` triggers the interactive approval prompt, and only
    when the container actually carries a key value. Everything here fails
    open to ``None``: no provider, no resolvable config, no key — no seed.
    """
    config_path = render_dir / "config.yml"
    if not config_path.is_file():
        return None
    try:
        import yaml

        from osprey.build.claude_code_resolver import ClaudeCodeModelResolver

        config = yaml.safe_load(config_path.read_text()) or {}
        spec = ClaudeCodeModelResolver.resolve(
            config.get("claude_code") or {},
            (config.get("api") or {}).get("providers") or {},
            include_telemetry=False,
        )
    except Exception as exc:  # noqa: BLE001 — the other seeds must still land
        logger.warning("Could not resolve the provider spec from %s (%r)", config_path, exc)
        return None
    if spec is None or spec.auth_env_var != _PROMPTED_AUTH_ENV_VAR:
        return None
    key = env.get(spec.auth_secret_env or _PROMPTED_AUTH_ENV_VAR, "").strip()
    if not key:
        return None
    return key[-_API_KEY_DIGEST_CHARS:]


def _hand_ownership(target: Path, base_dir: Path, owner_user: str | None) -> None:
    """Chown the seed's writes to the privilege-drop target, when root wrote them.

    Mirrors the entrypoint's state-zone hand-back: any root-phase step that
    writes where the dropped process must later write hands its files over
    before the drop. The directory is handed over too (non-recursively, like
    the CLAUDE.md seeder's chown of the same mount): Claude Code rewrites the
    state file through a tmp + rename, and a rename needs write on the
    DIRECTORY — a fresh named volume mounts root-owned, so an osprey-owned
    file in a root-owned directory would still be one Claude Code cannot
    update. A no-op when unprivileged (tests, ``--user`` starts) or when no
    owner was named.
    """
    if not owner_user or os.name != "posix" or os.geteuid() != 0:
        return
    try:
        import pwd

        record = pwd.getpwnam(owner_user)
        os.chown(target, record.pw_uid, record.pw_gid)
        os.chown(base_dir, record.pw_uid, record.pw_gid)
    except (KeyError, OSError) as exc:
        logger.warning(
            "Could not hand %s to user %r (%s); the session may be unable to update it.",
            target,
            owner_user,
            exc,
        )

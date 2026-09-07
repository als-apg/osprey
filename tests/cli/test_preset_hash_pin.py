"""Pin: every bundled preset's resolved-content hash.

``preset_hash`` is stamped into ``.osprey-manifest.json`` at build time and
compared by the deploy-side staleness advisory, so a hash that moves reports
drift on every already-deployed project. The digests below are deliberately
hardcoded rather than recomputed — a test that derives its expectation from the
code under test would pin nothing.

Any change to these values means a preset's *resolved content* changed. That is
a legitimate thing to do, but it is a deploy-visible event: update the digest
here in the same commit, knowingly.
"""

from __future__ import annotations

from osprey.cli.build_profile import list_presets
from osprey.cli.build_profile_merge import _hash_resolved_profile, compute_preset_hash

# preset name -> resolved-content hash.
PINNED_PRESET_HASHES: dict[str, str] = {
    # Every digest moved together when the app templates were converted into the
    # presets. A preset now carries the whole declarative config a deployment
    # renders — the control system, the services, the approval table, the panel
    # selection — where before that content lived in `apps/<name>/config.yml.j2`
    # and was rendered underneath the profile. `app_template:` is also popped
    # before hashing now, so it contributes nothing to a digest. Every already
    # deployed project therefore reads stale exactly once, which is the correct
    # signal: its rebuilt config really is written from a different source.
    #
    # Merging main moved them all a second time, and for a separate reason:
    # `turn-state` joined every preset's `hooks:`, shipping the reporter that
    # tells the web terminal when a turn starts and ends. A rebuilt project
    # gains the hook file in `.claude/hooks/` and four settings.json
    # registrations (UserPromptSubmit, Stop, StopFailure, and a SessionStart
    # entry matched to `startup|resume|clear`). Deploy-visible, so the
    # staleness advisory firing on already-deployed projects is correct.
    # The third move, one commit: `control-context` joined every preset's
    # `hooks:`, putting the active control target and each target's write state
    # in front of the agent at session start and whenever it changed since its
    # last turn. A rebuilt project gains the hook file in `.claude/hooks/` and
    # two settings.json registrations (SessionStart, UserPromptSubmit).
    "ariel-standalone": ("sha256:99e429dc923ba5276672f3c18857976ae517e3e05ef38f21f2fc7b72fad9386c"),
    "channel-finder-standalone": (
        "sha256:5c5d670fb6d048e854dfa5ecff9b02d6b9bf504a4cd574d6a5f66b34e18e3fa1"
    ),
    "control-assistant": (
        "sha256:a35bc6778ca15ff0e11b680b894d2b5488e9c753b9fa901cd8f26313490e6554"
    ),
    "control-assistant-admin": (
        "sha256:4f1496e3921e32937cad756be25c639061abfa62149527380fb513f480c79e6f"
    ),
    "control-assistant-knowledge": (
        "sha256:5ef269fd5090e53a98c52b9a8f7a9be0837e48e70f8f8cbc109d4e372c42d025"
    ),
    "control-assistant-logbook": (
        "sha256:0154037230bcfbecac1bcb11e1823afb0f65a328fee7931c8985e8069b72f243"
    ),
    "control-assistant-readonly": (
        "sha256:5f751306dd38faed68eaacc69e1bb8ce1dbd0fa79b5a739c9d796a6e7d735f41"
    ),
    "control-assistant-readwrite": (
        "sha256:519e9cfb344b95f6bb094057704b2ea5fe620a997d82a5a24da82ec74fa90003"
    ),
    "control-assistant-va-readwrite": (
        "sha256:2b74afdad052c055cfa12c682815d0d7a771fd5723826c845ecbbf86ed398f23"
    ),
    "hello-world": ("sha256:495e21e03af8f135964a3ebffa141e7dc6061183fa5d00978b81cd4ab3657e6a"),
}


def test_bundled_preset_set_is_pinned():
    """A new preset must be classified here before it ships.

    Without this the digest comparison below would silently skip an unpinned
    preset, and the pin would degrade as presets are added.
    """
    assert list_presets() == sorted(PINNED_PRESET_HASHES)


def test_every_bundled_preset_hash_is_unchanged():
    """Every preset resolves to its pinned digest.

    This is what gives the values above their meaning: without it the dict is
    dead data, and a preset's resolved content could move — a deploy-visible
    event every already-deployed project reads as stale — with nothing saying
    so.
    """
    actual = {name: compute_preset_hash(name) for name in list_presets()}
    assert actual == PINNED_PRESET_HASHES


def test_hashing_does_not_mutate_the_callers_dict(tmp_path):
    """The caller's dict survives hashing unchanged.

    ``_hash_resolved_profile`` resolves ``extends`` and folds in file material
    to build what it digests, and every caller keeps using the dict it passed
    in afterwards.
    """
    raw = {"name": "Demo", "provider": "anthropic", "model": "haiku"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "provider": "anthropic", "model": "haiku"}

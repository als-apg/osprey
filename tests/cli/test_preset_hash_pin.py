"""Pin: the YAML-surface rename must not move any bundled preset's hash.

``preset_hash`` is stamped into ``.osprey-manifest.json`` at build time and
compared by the deploy-side staleness advisory, so a hash that moves for a
purely cosmetic key rename would report drift on every already-deployed
project. The digests below were recorded from the bundled presets *before* the
``data_bundle`` → ``app_template`` YAML rename; they are the durable evidence
that the rename stayed hash-neutral, and they are deliberately hardcoded rather
than recomputed — a test that derives its expectation from the code under test
would pin nothing.

Any change to these values means a preset's *resolved content* changed. That is
a legitimate thing to do, but it is a deploy-visible event: update the digest
here in the same commit, knowingly.
"""

from __future__ import annotations

from osprey.cli.build_profile import compute_preset_hash, compute_profile_hash, list_presets
from osprey.cli.build_profile_merge import _hash_resolved_profile

# preset name -> resolved-content hash, pre-rename.
PINNED_PRESET_HASHES: dict[str, str] = {
    "ariel-standalone": "sha256:a08bde688f81f7604da07822db5def68d6c0d294688f15ec8720ac5df11a8cee",
    "channel-finder-standalone": (
        "sha256:9faa42d633aae7917429c3ec327c004672e68fa7617832d5ae245780fcb2a20f"
    ),
    # A digest here is the resolved content of the preset AND of every preset
    # that extends it, so a change in a base moves all three control-assistant
    # entries together. Comment rewrites cannot move a digest
    # (`_hash_resolved_profile` hashes resolved canonical JSON and says so);
    # only a key or value change can. Some such changes are behavior-neutral
    # and some are not — a rebuilt project can gain or lose a tab, a hook, a
    # health check or a skill directory — and in either case the deploy-side
    # staleness advisory firing on already-deployed projects is the correct
    # signal, not noise.
    "control-assistant": "sha256:025fa1e628849cc7d162af29bb4f95eb41668f7e6eb9ab751b58238c00d8008e",
    "control-assistant-readonly": (
        "sha256:7c687a478ab8ab5e928a21696244f21931c9699f7fc5fbf698b69a3ca8438c6d"
    ),
    "control-assistant-readwrite": (
        "sha256:612b576a25097949157b55e4b31122a30db16266d866e45e0f0379f3385ead3e"
    ),
    "hello-world": "sha256:ac9c00d70922c3c88d561f7ffa29af3ccb1650d5a8bfaa13b884563199ce371a",
}


def test_bundled_preset_set_is_pinned():
    """A new preset must be classified here before it ships.

    Without this the per-preset loop below would silently skip an unpinned
    preset, and the pin would degrade as presets are added.
    """
    assert list_presets() == sorted(PINNED_PRESET_HASHES)


def test_every_bundled_preset_hash_is_unchanged():
    """Every preset resolves to its pre-rename digest — the rename is hash-neutral."""
    actual = {name: compute_preset_hash(name) for name in list_presets()}
    assert actual == PINNED_PRESET_HASHES


def test_hash_is_neutral_to_the_yaml_surface_spelling(tmp_path):
    """Both spellings of the same profile hash identically.

    Exercises ``_hash_resolved_profile`` with dicts that did NOT pass through
    ``_read_profile_document``, which is the only way the YAML-surface spelling
    can still reach it — and the case the canonicalization exists for.
    """
    profile_path = tmp_path / "profile.yml"
    yaml_spelling = {"name": "Demo", "app_template": "hello_world"}
    field_spelling = {"name": "Demo", "data_bundle": "hello_world"}

    assert _hash_resolved_profile(yaml_spelling, profile_path) == _hash_resolved_profile(
        field_spelling, profile_path
    )


def test_hash_still_tracks_the_bundle_value(tmp_path):
    """Canonicalizing spellings must not collapse *different* bundles to one hash."""
    profile_path = tmp_path / "profile.yml"
    assert _hash_resolved_profile(
        {"name": "Demo", "app_template": "hello_world"}, profile_path
    ) != _hash_resolved_profile({"name": "Demo", "app_template": "ariel_standalone"}, profile_path)


def test_hashing_does_not_mutate_the_callers_dict(tmp_path):
    """The caller's dict survives hashing with its own spelling intact."""
    raw = {"name": "Demo", "app_template": "hello_world"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "app_template": "hello_world"}


def test_normalization_happens_before_extends_resolution(tmp_path):
    """Pins the *placement* of the canonicalization, not just its presence.

    A mixed-spelling chain — parent on disk in the canonical spelling, caller
    dict in the YAML spelling with a *different* value — only merges child-wins
    if ``raw`` is canonicalized before ``_resolve_extends``. Normalize after the
    merge instead and the resolved dict carries both keys with conflicting
    values, which raises; both public entry points swallow that to ``None``,
    so the build would stamp no ``preset_hash`` and deploy-side staleness would
    silently stop comparing rather than fail loudly.

    The plain neutrality test above passes under either placement, so this is
    the one that holds the line.
    """
    (tmp_path / "parent.yml").write_text(
        "name: Parent\ndata_bundle: ariel_standalone\n", encoding="utf-8"
    )
    child_path = tmp_path / "child.yml"

    mixed = {"extends": "parent.yml", "name": "Child", "app_template": "hello_world"}
    canonical = {"extends": "parent.yml", "name": "Child", "data_bundle": "hello_world"}
    assert _hash_resolved_profile(mixed, child_path) == _hash_resolved_profile(
        canonical, child_path
    )

    # The same chain through the public entry point must produce a hash, never
    # the "cannot compare" None that a raise would collapse to.
    child_path.write_text(
        "extends: parent.yml\nname: Child\napp_template: hello_world\n", encoding="utf-8"
    )
    assert compute_profile_hash(child_path) is not None

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
    # that extends it, so a change in a base moves all four control-assistant
    # entries together. Last moved when the bluesky-web sidecar shed its
    # panels-era name, a deploy-visible change (the panel's backing service
    # and its config keys are renamed). Comment rewrites cannot move a digest
    # (`_hash_resolved_profile` hashes resolved canonical JSON and says so);
    # only a key or value change can. Some such changes are behavior-neutral
    # and some are not — a rebuilt project can gain or lose a tab, a hook, a
    # health check or a skill directory — and in either case the deploy-side
    # staleness advisory firing on already-deployed projects is the correct
    # signal, not noise.
    # Moved when the control-assistant preset turned password login on for its
    # web terminals (auth stanza, ariel's `login: false`, demo passwords under
    # `env.defaults`). All four moved together, as the note above predicts.
    # Moved again when pymongo became a core OSPREY dependency and the preset
    # dropped its `dependencies: [pymongo>=4.0]` line. Behavior-neutral for a
    # rebuilt project — pymongo still lands in its venv, now from the base
    # install rather than the profile — but the generated pyproject.toml no
    # longer names it, so the advisory firing is correct.
    "control-assistant": "sha256:7410d74d603fefa04d8ad79e042782da32b01a9acb0ca8871786e93ad3d65f13",
    "control-assistant-ariel": (
        "sha256:9e71eaa1c45afcf201a12a1722fab5b80ffcb059ff45b0d602785c7e37a1cea5"
    ),
    # The two operator tiers below moved together, and alone, when each gained
    # the single dotted key `services.graphdb.port_host: 7687` in its `config:`
    # block — the attached-render personas scaffold no services of their own, so
    # without it their terminals would dial the shipped default port rather than
    # the port the hosting deployment publishes its graph store on. The base
    # `control-assistant` and the `control-assistant-ariel` tier are untouched
    # (the change is in these two leaves, not in the base they extend), which is
    # why only two of the four digests above move here. NOT behavior-neutral: a
    # rebuilt operator terminal gains the `graph` MCP server and its tools, so
    # the deploy-side staleness advisory firing on already-deployed
    # operator-tier projects is the correct signal.
    "control-assistant-readonly": (
        "sha256:7ecd21a4eee57c7aa370f9434538007c835d81d457b13a8b067c396af8c79ac7"
    ),
    "control-assistant-readwrite": (
        "sha256:17f8d92db053ce23154e0a97f42902223fbd84397e735a32534ba5255b64670f"
    ),
    # Moved when the onboarding rewrite dropped the `facility` rule. The
    # wholesale comment rewrite that shipped alongside it contributed nothing:
    # the digest is comment-blind, so the rule drop is the entire delta.
    "hello-world": "sha256:9a87fb40f03287e8b4605c63a56cb8220ae1c32cd2e8c7df9a7d01ad370f358a",
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

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
    "ariel-standalone": "sha256:a401c62979ff477c8294954a39c15076f3b6d981826f2ec01ffbe863ae23e9e5",
    "channel-finder-standalone": (
        "sha256:03c3d35070730762db3d46ef88beabe3b0e0bf98abd5fff72d81a95d2b6fc918"
    ),
    # The web-terminal presets were re-pinned when the shipped roster went
    # from a bare-string first user to fully explicit name/index/persona
    # entries — a resolved-content change (deploy-visible as staleness), made
    # knowingly per the module docstring. The resolved *behavior* is identical.
    #
    # Re-pinned again when control-assistant's control_system.type default
    # flipped from "mock" to "virtual_accelerator" (mock is now the documented
    # fallback via `osprey set connector=mock`) — both extends
    # children inherit the new default, so all three digests moved.
    #
    # Re-pinned again when the RESULTS panel became BLUESKY: the preset's
    # web_panels entry and its web.panels.<id>.{label,url,path} overrides all
    # moved from `results` to `bluesky`. Unlike the two re-pins above this one
    # is NOT behavior-neutral — a rebuilt project gets a differently-named tab
    # — so the staleness advisory firing on already-deployed projects is the
    # correct signal, not noise. Both extends children inherit it, so all three
    # digests moved again.
    #
    # Re-pinned again when control-assistant gained the test-ioc-safety rule
    # selection (renders only for EPICS-family control systems). Both extends
    # children inherit it, so all three digests moved.
    #
    # Re-pinned again for the lifecycle redesign. The preset's diff is almost
    # entirely comment text — which this hash cannot see, because
    # `_hash_resolved_profile` hashes resolved canonical JSON and says so. The
    # one CONTENT change is the two persona `project_path` values, which moved
    # from the retired sibling layout (`../control-assistant-<persona>`) into
    # the render zone (`build/control-assistant-<persona>`). Both extends
    # children inherit it, so all three digests moved.
    "control-assistant": "sha256:645526195bf195431794e55e26aedb80a5ce6320e59931ac8a31b6bc786ad63f",
    "control-assistant-readonly": (
        "sha256:097d786bfaa8a57199959f76192f0b12676b216714203f227fdd7aa9a5f6bc2c"
    ),
    "control-assistant-readwrite": (
        "sha256:b8bd051c74142cf20224333609c9fb56012e5fb7f15fef3bcbac80eaa349df13"
    ),
    "hello-world": "sha256:e1666b0b1a1d1232bc3aa9c32ccf11e3555a217162fda292f4240396ef19ec8a",
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

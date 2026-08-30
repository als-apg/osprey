"""``anchored_put(..., comment=...)`` — a written key explains itself in place.

A caller that writes a key into a rendered config can hand ``comment=`` a note
explaining, in the file, why that key is there. That note has to land above
the key without disturbing the template prose around it, and a rebuild has to
replace the note rather than stack a second copy on it. The suite below
exercises that against a ``target_switch`` block shaped like the shipped
template — banner prose, wrapped inline comments on neighboring keys, a
commented-out example, and a following top-level section — using a written
``live_gateway_acknowledged`` key as the vehicle, since that shape puts every
hazard the split has to survive in one block.

The awkward part is where ruamel keeps such blocks. A round-trip load gives a
key no "comment above me" slot of its own unless it is the mapping's first: a
standalone block is read back appended to the *previous* key's end-of-line
comment token — the same token that holds that key's inline comment and every
continuation line the inline comment wraps onto. In the shipped ``target_switch``
block one token therefore holds three unrelated things at once:
``probe_interval_s``'s wrapped inline comment, the commented-out example that
closes the block, and the ``# Archiver Configuration`` banner that opens the
next top-level section. Splitting that token in the wrong place tears the inline
comment in half; not splitting it at all strands the new key below the archiver
banner. The tests below pin the split that is correct on all three.
"""

from __future__ import annotations

import io

from ruamel.yaml import YAML

from osprey.utils.config_writer import anchored_put

# Cut from the shipped template (control_assistant/config.yml.j2): the
# acknowledgment prose is part of the header above `target_switch:`, the two
# scalars carry wrapped inline comments, the commented-out example closes the
# block, and a top-level section follows. `live_gateway_acknowledged` is not
# written by the build here — it's this fixture's stand-in key, chosen because
# its surrounding block exercises every neighboring-comment hazard at once.
TARGET_SWITCH = """\
control_system:
  # Control-System Target Switch (Layer 2)
  # Bounds how a running session moves between the connector targets above.
  #
  # The `live_gateway_acknowledged` key below is the operator acknowledgment for
  # the live machine. Set it to your own live gateway's hostname to confirm the
  # `epics` gateways above really are your facility's.
  target_switch:
    drain_timeout_s: 5      # Seconds in-flight operations get to finish on the
                            # old target before it is torn down regardless
    probe_interval_s: 30    # Seconds between background reachability probes of
                            # every target's gateways
    # live_gateway_acknowledged: your-ca-gateway.example.com

# Archiver Configuration
archiver:
  type: mock_archiver
"""

PLAIN = """\
services:
  postgresql:
    path: ./services/postgresql
  openobserve:
    path: ./services/openobserve
    port: 5080          # Host port

# Services to deploy with `osprey up`
deployed_services:
  - postgresql
"""

NOTE = "Written by osprey build for the stand-in.\nReplace by hand when going live."


def _yaml() -> YAML:
    """A round-trip handler configured exactly like the writer's own."""
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _round_trip(text: str, mutate) -> str:
    yaml = _yaml()
    data = yaml.load(text)
    mutate(data)
    out = io.StringIO()
    yaml.dump(data, out)
    return out.getvalue()


def _inject(data) -> None:
    anchored_put(
        data["control_system"]["target_switch"],
        "live_gateway_acknowledged",
        "localhost:5074",
        comment=NOTE,
    )


class TestCommentPlacement:
    def test_the_injected_block_renders_exactly_like_this(self):
        # The whole file, pinned: it is the only assertion that also proves
        # nothing moved and nothing was duplicated anywhere else.
        assert _round_trip(TARGET_SWITCH, _inject) == (
            "control_system:\n"
            "  # Control-System Target Switch (Layer 2)\n"
            "  # Bounds how a running session moves between the connector targets above.\n"
            "  #\n"
            "  # The `live_gateway_acknowledged` key below is the operator acknowledgment for\n"
            "  # the live machine. Set it to your own live gateway's hostname to confirm the\n"
            "  # `epics` gateways above really are your facility's.\n"
            "  target_switch:\n"
            "    drain_timeout_s: 5      # Seconds in-flight operations get to finish on the\n"
            "                            # old target before it is torn down regardless\n"
            "    probe_interval_s: 30    # Seconds between background reachability probes of\n"
            "                            # every target's gateways\n"
            "    # Written by osprey build for the stand-in.\n"
            "    # Replace by hand when going live.\n"
            "    live_gateway_acknowledged: localhost:5074\n"
            "    # live_gateway_acknowledged: your-ca-gateway.example.com\n"
            "\n"
            "# Archiver Configuration\n"
            "archiver:\n"
            "  type: mock_archiver\n"
        )

    def test_root_level_comment_renders_flush_left(self):
        text = _round_trip(PLAIN, lambda data: anchored_put(data, "facility", 1, comment="Note"))

        assert "# Note\nfacility: 1\n" in text


class TestNeighbouringComments:
    """The two failure shapes the split has to avoid, named one at a time."""

    def test_previous_keys_wrapped_inline_comment_is_not_split(self):
        text = _round_trip(TARGET_SWITCH, _inject)

        # What a plain anchored_put does here: it treats everything past the
        # first line of probe_interval_s's token as a detachable block, so the
        # new key lands between the inline comment and the continuation line
        # that finishes the sentence.
        assert (
            "    probe_interval_s: 30    # Seconds between background reachability probes of\n"
            "                            # every target's gateways\n"
            "    # Written by osprey build for the stand-in.\n"
        ) in text
        assert (
            "    drain_timeout_s: 5      # Seconds in-flight operations get to finish on the\n"
            "                            # old target before it is torn down regardless\n"
        ) in text

    def test_the_next_sections_banner_stays_below_the_new_key(self):
        text = _round_trip(TARGET_SWITCH, _inject)

        # What a naive `mapping[key] = value` does here: the banner opening the
        # next top-level section is stored on this block's last key, so the new
        # entry renders after it — visually inside `archiver:`, and below a
        # header that has nothing to do with it.
        assert text.index("live_gateway_acknowledged: localhost:5074") < text.index(
            "# Archiver Configuration"
        )
        assert text.index("# Archiver Configuration") < text.index("archiver:")

    def test_the_template_prose_and_example_survive_exactly_once(self):
        text = _round_trip(TARGET_SWITCH, _inject)

        assert text.count("# The `live_gateway_acknowledged` key below is the operator") == 1
        assert text.count("# live_gateway_acknowledged: your-ca-gateway.example.com") == 1
        assert text.count("live_gateway_acknowledged: localhost:5074") == 1
        assert text.count("# Written by osprey build for the stand-in.") == 1
        # The header prose keeps introducing the block it documents.
        assert text.index("# The `live_gateway_acknowledged` key below") < text.index(
            "  target_switch:"
        )

    def test_a_wrapped_inline_comment_with_nothing_after_it_is_left_alone(self):
        # The complement of the split: when the last entry's token is nothing
        # but a wrapped inline comment there is no block to move, and the
        # continuation must not be mistaken for one.
        doc = (
            "block:\n"
            "  first: 1        # a comment that wraps onto\n"
            "                  # a second line and stops\n"
        )
        text = _round_trip(doc, lambda data: anchored_put(data["block"], "second", 2, comment="Hi"))

        assert text == (
            "block:\n"
            "  first: 1        # a comment that wraps onto\n"
            "                  # a second line and stops\n"
            "  # Hi\n"
            "  second: 2\n"
        )


class TestIdempotence:
    def test_rewriting_the_same_key_reproduces_the_file_byte_for_byte(self):
        once = _round_trip(TARGET_SWITCH, _inject)
        twice = _round_trip(once, _inject)

        assert twice == once

    def test_rewriting_a_first_key_reproduces_the_file_byte_for_byte(self):
        # The first key of a mapping is the one case where ruamel keeps the
        # block on the mapping itself rather than on a predecessor, so the
        # replacement path differs and needs its own pin.
        doc = "outer:\n  inner:\n    seed: 1\n"

        def mutate(data):
            anchored_put(data["outer"]["inner"], "seed", 1, comment="Kept in step by the build")

        once = _round_trip(doc, mutate)
        twice = _round_trip(once, mutate)

        assert "    # Kept in step by the build\n    seed: 1\n" in once
        assert twice == once

    def test_a_reworded_note_is_added_beside_the_old_one(self):
        # The boundary of the replacement rule, pinned deliberately. Once
        # written, a note is indistinguishable from the template prose around
        # it — both are plain comment lines on the same token at the same
        # indent. Only a run matching the text about to be written is removed,
        # so re-running one build is idempotent while a build whose wording
        # changed leaves the old wording for a human to remove. Guessing wider
        # would mean deleting template prose.
        once = _round_trip(TARGET_SWITCH, _inject)
        twice = _round_trip(
            once,
            lambda data: anchored_put(
                data["control_system"]["target_switch"],
                "live_gateway_acknowledged",
                "localhost:5074",
                comment="A different note entirely",
            ),
        )

        assert "# A different note entirely" in twice
        assert twice.count("# Written by osprey build for the stand-in.") == 1
        assert twice.count("live_gateway_acknowledged: localhost:5074") == 1


class TestWithoutComment:
    """``comment=None`` must leave the pre-existing behavior untouched."""

    def test_nested_and_root_puts_render_as_before(self):
        def mutate(data):
            anchored_put(data["services"], "virtual_accelerator", {"port": 5064})
            anchored_put(data, "facility", {"prefix": "ca"}, comment=None)

        assert _round_trip(PLAIN, mutate) == (
            "services:\n"
            "  postgresql:\n"
            "    path: ./services/postgresql\n"
            "  openobserve:\n"
            "    path: ./services/openobserve\n"
            "    port: 5080          # Host port\n"
            "  virtual_accelerator:\n"
            "    port: 5064\n"
            "\n"
            "# Services to deploy with `osprey up`\n"
            "deployed_services:\n"
            "  - postgresql\n"
            "facility:\n"
            "  prefix: ca\n"
        )

    def test_plain_dict_target_still_takes_a_comment_request(self):
        # Nowhere to hang a comment on a plain dict; the assignment still happens.
        plain: dict = {}
        anchored_put(plain, "key", "value", comment="ignored")
        assert plain == {"key": "value"}

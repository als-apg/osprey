"""``control_system.writes`` takes exactly the strings ``never``, ``asked`` and ``allowed``.

Every layer is checked at parse, after the retired write keys are rewritten to
the tier key, so a normalized document is checked too. Anything else — a
boolean, a differently-cased string, ``yes``, ``null``, a list — is refused
with an error naming the document and the three values. Whether a login may
spell an arming word is its audience's question (a login ``shared:`` names may
not), judged where its render is composed. The ``--set`` layer is checked the
same way.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from osprey.cli import build_profile_document as doc
from osprey.cli.build_profile_document import _parse_profile_document, _read_profile_document
from osprey.cli.build_profile_merge import _judge_shared_posture, render_document
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.utils.workspace import TIERS_DIRNAME
from osprey_connectors.errors import BuildProfileError


@pytest.fixture(autouse=True)
def _fresh_state():
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()


#: Invalid values, as YAML text.
_INVALID = {
    "true": "true",
    "capitalised": '"Allowed"',
    "yes": '"yes"',
    "retired-on": '"on"',
    "null": "null",
    "list": "[allowed]",
}

#: Where a document lives decides its kind.
_KINDS = {
    "root": None,
    "tier": TIERS_DIRNAME,
}


def _write(tmp_path: Path, directory: str | None, text: str) -> Path:
    folder = tmp_path if directory is None else tmp_path / directory
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("profile.yml" if directory is None else "doc.yml")
    target.write_text(textwrap.dedent(text), encoding="utf-8")
    return target


def _assert_names_values(message: str, source: str) -> None:
    assert source in message
    assert "`control_system.writes`" in message
    assert "`allowed`" in message
    assert "`asked`" in message
    assert "`never`" in message


@pytest.mark.parametrize("kind", sorted(_KINDS))
@pytest.mark.parametrize("label", sorted(_INVALID))
def test_an_invalid_value_is_refused_nested(tmp_path, kind, label):
    path = _write(
        tmp_path,
        _KINDS[kind],
        f"""
        config:
          control_system:
            writes: {_INVALID[label]}
        """,
    )
    with pytest.raises(BuildProfileError) as excinfo:
        _read_profile_document(path, source="my-doc.yml")
    _assert_names_values(str(excinfo.value), "my-doc.yml")


@pytest.mark.parametrize("kind", sorted(_KINDS))
@pytest.mark.parametrize("label", sorted(_INVALID))
def test_an_invalid_value_is_refused_dotted(tmp_path, kind, label):
    path = _write(
        tmp_path,
        _KINDS[kind],
        f"""
        config:
          control_system.writes: {_INVALID[label]}
        """,
    )
    with pytest.raises(BuildProfileError) as excinfo:
        _read_profile_document(path, source="my-doc.yml")
    _assert_names_values(str(excinfo.value), "my-doc.yml")


def test_the_refusal_shows_the_value_given(tmp_path):
    path = _write(tmp_path, None, 'config:\n  control_system:\n    writes: "yes"\n')
    with pytest.raises(BuildProfileError) as excinfo:
        _read_profile_document(path, source="p.yml")
    assert "'yes'" in str(excinfo.value)


def test_an_invalid_value_in_any_spelling_is_refused(tmp_path):
    """A dotted value is checked even when a later nested one would win the render."""
    path = _write(
        tmp_path,
        None,
        """
        config:
          control_system.writes: maybe
          control_system:
            writes: never
        """,
    )
    with pytest.raises(BuildProfileError):
        _read_profile_document(path, source="p.yml")


@pytest.mark.parametrize("kind", ["root", "tier"])
@pytest.mark.parametrize("value", ["allowed", "asked", "never"])
def test_the_three_values_are_accepted(tmp_path, kind, value):
    path = _write(
        tmp_path,
        _KINDS[kind],
        f"""
        config:
          control_system:
            writes: {value}
        """,
    )
    raw = _read_profile_document(path, source="p.yml")
    assert raw["config"]["control_system"]["writes"] == value


@pytest.mark.parametrize("word", ["allowed", "asked"])
@pytest.mark.parametrize(
    "text",
    [
        "config:\n  control_system:\n    writes: {word}\n",
        "config:\n  control_system.writes: {word}\n",
    ],
)
def test_a_login_may_spell_an_arming_word_at_parse(tmp_path, text, word):
    """The parse point reads the tier; a shared login's seal refuses it later."""
    path = _write(tmp_path, TIERS_DIRNAME, text.format(word=word))
    raw = _read_profile_document(path, source="tiers/viewer.yml")
    assert doc.validate_tier_key_value(raw, "tiers/viewer.yml") is None


@pytest.mark.parametrize("word", ["allowed", "asked"])
@pytest.mark.parametrize("key", ["control_system.writes", "control_system"])
def test_the_shared_composer_refuses_an_arming_word_itself(word, key):
    """A shared login's document reaches the composer's own door, with its refusal."""
    config = {key: word if key == "control_system.writes" else {"writes": word}}
    with pytest.raises(BuildProfileError) as excinfo:
        _judge_shared_posture(config, "tiers/viewer.yml", None)
    message = str(excinfo.value)
    assert "tiers/viewer.yml" in message
    assert f"cannot spell `control_system.writes: {word}`" in message
    assert "`control_system.writes: never`" in message
    assert "modules.web_terminals.shared" in message


@pytest.mark.parametrize("value", ["never", "on", None])
def test_the_shared_composer_passes_what_arms_nothing(value):
    _judge_shared_posture({"control_system.writes": value}, "tiers/viewer.yml", None)


def test_a_normalized_login_reads_its_retired_answer(tmp_path):
    """The retired keys in a login normalize to the answer they give, which the check accepts."""
    path = _write(
        tmp_path,
        TIERS_DIRNAME,
        "config:\n  control_system:\n    writes_enabled: true\n",
    )
    raw = _read_profile_document(path, source="tiers/viewer.yml")
    assert raw["config"]["control_system"]["writes"] == "allowed"


def test_a_normalized_tier_reads_allowed_and_passes(tmp_path):
    path = _write(
        tmp_path,
        TIERS_DIRNAME,
        "config:\n  control_system.writes_enabled: true\n",
    )
    raw = _read_profile_document(path, source="tier.yml")
    assert raw["config"]["control_system.writes"] == "allowed"


def test_a_silent_document_is_untouched():
    raw = _parse_profile_document("config:\n  control_system:\n    type: mock\n", "p.yml")
    assert raw == {"config": {"control_system": {"type": "mock"}}}


def test_a_document_without_a_mapping_config_passes():
    assert _parse_profile_document("config: null\n", "p.yml") == {"config": None}
    assert _parse_profile_document("name: x\n", "p.yml") == {"name": "x"}


def test_a_text_document_is_read_as_a_root():
    """No file: the document is a root, so an arming word is fine."""
    for word in ("allowed", "asked"):
        raw = _parse_profile_document(f"config:\n  control_system.writes: {word}\n", "text")
        assert raw["config"]["control_system.writes"] == word


@pytest.mark.parametrize("value", ["yes", "true", "Allowed", "Asked", "null", "[allowed]"])
def test_an_invalid_value_on_the_set_layer_is_refused(value):
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None, "hello-world", set_pairs=(f"config.control_system.writes={value}",)
        )
    _assert_names_values(str(excinfo.value), "--set")


@pytest.mark.parametrize("value", ["allowed", "asked", "never"])
def test_a_valid_value_on_the_set_layer_is_accepted(value):
    resolve_build_profile(None, "hello-world", set_pairs=(f"config.control_system.writes={value}",))


@pytest.mark.parametrize(
    "text",
    [
        "config:\n  images:\n    platform: linux/amd64\n",
        "config:\n  images.platform: linux/amd64\n",
        "lists: own\nconfig:\n  images.platform: linux/arm64\n",
    ],
)
def test_a_login_naming_the_image_platform_is_refused_naming_the_profile(tmp_path, text):
    """The images are the deployment's: one platform, stated in its profile."""
    path = _write(tmp_path, TIERS_DIRNAME, text)
    layer = _read_profile_document(path, source=f"{TIERS_DIRNAME}/x.yml")

    with pytest.raises(BuildProfileError) as excinfo:
        render_document({"name": "host"}, layer, f"{TIERS_DIRNAME}/x.yml")

    message = str(excinfo.value)
    assert f"{TIERS_DIRNAME}/x.yml" in message
    assert "images.platform" in message and "profile.yml" in message


def test_a_login_naming_another_image_key_is_not_refused_for_it(tmp_path):
    path = _write(tmp_path, TIERS_DIRNAME, "config:\n  images.tag: dev\n")
    layer = _read_profile_document(path, source=f"{TIERS_DIRNAME}/x.yml")

    render_document({"name": "host"}, layer, f"{TIERS_DIRNAME}/x.yml")

"""The picture-search guide states the values the code and the packaged files hold.

``docs/source/how-to/ariel/picture-search.rst`` documents the site-run
llama-server command, the format table, the ``llama-cpp`` provider entry, the
measured values and the upgrade notes. Each pinned fact here is compared
against its source of truth (the format registry, the presets, the packaged
``providers.yml``, the CLI's ``--force`` refusal) rather than a literal copied
into the test, so a change on either side fails here first.

The measured latency gate failed for the uncapped command, so the documented
command carries ``--image-max-tokens 256``.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml

from osprey.imaging.formats import ROWS
from osprey.models.providers.llama_cpp import LLAMA_CPP_DEFAULT_MODEL
from osprey.services.ariel_search.cli_operations import FORCE_REFUSAL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source"
_GUIDE = _DOCS / "how-to" / "ariel" / "picture-search.rst"
_INDEX = _DOCS / "how-to" / "ariel" / "index.rst"
_INGESTION = _DOCS / "how-to" / "ariel" / "data-ingestion.rst"
_CONTRACT = _DOCS / "reference" / "contracts" / "ariel.rst"
_PROFILES = _REPO_ROOT / "src" / "osprey" / "profiles"
_PRESETS = ("control-assistant", "ariel-standalone")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """*text* with every whitespace run collapsed to one space."""
    return re.sub(r"\s+", " ", text)


def _code_blocks(text: str) -> list[str]:
    """The dedented bodies of every ``.. code-block::`` in *text*."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = re.match(r"^(\s*)\.\. code-block::", lines[i])
        if not match:
            i += 1
            continue
        indent = len(match.group(1))
        i += 1
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
            i += 1
        blocks.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return blocks


def _section(text: str, title: str) -> str:
    """The body of the ``=``-underlined section *title*, up to the next one."""
    match = re.search(rf"^{re.escape(title)}\n=+\n(.*?)(?=^[^\n]+\n=+\n|\Z)", text, re.M | re.S)
    assert match, f"section {title!r} missing from {_GUIDE.name}"
    return match.group(1)


def _command() -> str:
    commands = [b for b in _code_blocks(_text(_GUIDE)) if b.startswith("llama-server ")]
    assert len(commands) == 1, "the guide documents exactly one llama-server command"
    return commands[0]


def _preset_config(name: str) -> dict:
    data = yaml.safe_load(_text(_PROFILES / "presets" / f"{name}.yml"))
    return data["config"]


# -- the documented command ------------------------------------------------


def test_command_alias_is_the_presets_model() -> None:
    match = re.search(r"--alias (\S+)", _command())
    assert match, "the command carries --alias"
    alias = match.group(1)
    assert alias == "qwen3-vl-embedding-2b"
    assert alias == LLAMA_CPP_DEFAULT_MODEL
    for preset in _PRESETS:
        model = _preset_config(preset)["ariel.enhancement_modules.image_embedding.model"]
        assert alias == model, preset


def test_command_binds_to_localhost_and_reads_no_media_path() -> None:
    command = _command()
    assert "--host 127.0.0.1" in command
    assert "--media-path" not in command


def test_command_carries_the_image_token_cap() -> None:
    """The latency gate failed uncapped, so the cap is part of the command."""
    assert "--image-max-tokens 256" in _command()


def test_guide_cites_handle_media_without_condition() -> None:
    flat = _flat(_text(_GUIDE))
    assert "``b11277`` has no switch to disable remote image-URL fetch" in flat
    assert "``tools/server/server-common.cpp:1088-1102``" in flat
    assert "User-Agent: llama-cpp/b1-eae11d2" in flat


def test_guide_pins_build_and_weights() -> None:
    flat = _flat(_text(_GUIDE))
    for fact in (
        "--branch b11277",
        "eae11d2217fe9225d1aaba48773b6cca45ae4de9",
        "PR #29556",
        "DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF",
        "6a1b927414664e0e17dd379913e3416a1ae1b48d",
        "42a4ebc629ecc6514649e12b1529b857f54900273bb854f853c970fb90edd09d",
        "3f89a7768ffa6606935319f71bf56bb71871249ba549bf1080a0caea7a088613",
    ):
        assert fact in flat, fact


def test_host_networking_sentence() -> None:
    flat = _flat(_text(_GUIDE))
    assert (
        "A localhost-bound server is reachable only from containers on host networking. "
        "Run ``ariel-sync`` with host networking by setting "
        "``services.ariel_sync.network: host``, and do the same for any "
        "bridge-networked container that serves ``hybrid_search``; web terminals "
        "already use host networking."
    ) in flat


# -- the format table --------------------------------------------------------


def test_format_table_equals_the_registry() -> None:
    section = _section(_text(_GUIDE), "Picture formats")
    documented = re.findall(r"^\s+\* - ``(\w+)``\n\s+- (accepted|reserved)\b", section, re.M)
    assert documented == [(name, row.status) for name, row in ROWS.items()]


def test_format_table_mime_types_equal_the_registry() -> None:
    section = _flat(_section(_text(_GUIDE), "Picture formats"))
    for row in ROWS.values():
        for mime in row.mimes:
            assert f"``{mime}``" in section, mime


# -- the provider entry ------------------------------------------------------


def test_llama_cpp_stanza_equals_the_packaged_entry() -> None:
    packaged = yaml.safe_load(_text(_PROFILES / "providers.yml"))["providers"]["llama-cpp"]
    stanzas = [
        yaml.safe_load(block)
        for block in _code_blocks(_text(_GUIDE))
        if block.startswith("providers:") and "llama-cpp:" in block
    ]
    assert len(stanzas) == 1
    assert stanzas[0] == {"providers": {"llama-cpp": packaged}}


def test_profile_expand_named_only_with_its_side_effects() -> None:
    flat = _flat(_text(_GUIDE))
    assert (
        "``osprey profile expand --providers`` also writes it, with side effects: it fills "
        "in every key the profile leaves to its preset and stamps provenance, which turns "
        "on the strict preset-drift check"
    ) in flat


# -- upgrade notes ------------------------------------------------------------


def _upgrade_notes() -> str:
    return _flat(_section(_text(_GUIDE), "Upgrade notes"))


def test_upgrade_note_profile_keys_sentence() -> None:
    assert (
        "An existing deployment's ``profile.yml`` is explicit and does not gain the new "
        "keys; ``osprey validate`` lists them as drift. Add the lines shown below, or run "
        "``osprey profile expand`` (it writes every lacking leaf, with the same side-effect "
        "caveat as ``--providers`` below); until then ``image_caption`` and "
        "``image_embedding`` keep their code defaults (off) and the view tool its default (on)."
    ) in _upgrade_notes()


def test_upgrade_note_profile_lines_match_the_presets() -> None:
    section = _section(_text(_GUIDE), "Upgrade notes")
    blocks = [b for b in _code_blocks(section) if b.startswith("config:")]
    assert len(blocks) == 1
    shown = yaml.safe_load(blocks[0])["config"]
    for preset in _PRESETS:
        config = _preset_config(preset)
        for key, value in shown.items():
            assert config[key] == value, (preset, key)


def test_upgrade_note_count_query_is_the_fold_predicate() -> None:
    query = (
        "SELECT count(*) FROM enhanced_entries WHERE attachments @? "
        "'$[*] ? (@.caption != null && @.caption != \"\")'"
    )
    assert query in _code_blocks(_text(_GUIDE))


def test_upgrade_note_reembed_command_carries_limit() -> None:
    notes = _upgrade_notes()
    assert "osprey ariel enhance --module text_embedding --limit <N>" in notes


def test_upgrade_note_states_kept_v1_and_empty_list_limitation() -> None:
    notes = _upgrade_notes()
    assert "keeps the v1 indexes" in notes
    assert "attachment list becomes empty keeps its stored attachment rows" in notes


def test_upgrade_note_carries_the_force_refusal() -> None:
    assert FORCE_REFUSAL in _upgrade_notes()


def test_upgrade_note_purge_sentence() -> None:
    assert (
        "Changing ``--image-max-tokens`` or the model files changes every picture vector "
        "while the table name stays the same, so re-embed with "
        "``osprey ariel purge --embeddings-only`` followed by a catch-up (which re-embeds "
        "text too)."
    ) in _upgrade_notes()


# -- measurements ---------------------------------------------------------------

#: One row label per SC10 measurement, each with the value the guide states.
_SC10_ROWS = {
    "CPU time per picture": "3.77 s",
    "llama-server peak RSS": "7.57 GiB",
    "Query p95 under bulk": "0.94 s",
    "Render worker ``VmSize``": "30.6 MiB",
    "``qwen3-vl:4b`` with ``think:false``": "131.5 s",
    "Captions per hour at the defaults": "about 27",
    "Upgrade fold, 135,000 rows": "57.6 s",
    "v2 full-text index build, 135,000 rows": "27.2 s",
    "Trigram index build, 135,000 rows": "2.6 s",
    "Fusion calibration": "Uncalibrated",
}


def _measurement_rows() -> dict[str, str]:
    section = _section(_text(_GUIDE), "Measured values")
    rows = re.findall(r"^\s+\* - (.+?)\n\s+- (.+?)(?=^\s+\* - |\Z)", section, re.M | re.S)
    return {label.strip(): _flat(value) for label, value in rows}


@pytest.mark.parametrize(("label", "value"), sorted(_SC10_ROWS.items()))
def test_measurements_table_has_a_row_per_sc10_item(label: str, value: str) -> None:
    rows = _measurement_rows()
    matches = [v for k, v in rows.items() if k.startswith(label)]
    assert matches, f"no measurements row starting {label!r}"
    assert value in matches[0]


def test_measurements_carry_host_facts_and_latency_branch() -> None:
    section = _flat(_section(_text(_GUIDE), "Measured values"))
    for fact in (
        "``uname -m`` ``x86_64``",
        "AMD EPYC 7313 16-Core Processor",
        "64 logical CPUs",
        "503 GiB RAM",
        "docker host architecture ``x86_64``",
        "3.83 s without the cap (fails)",
        "0.94 s with ``--image-max-tokens 256`` (passes)",
        "The latency gate failed",
    ):
        assert fact in section, fact


def test_fusion_defaults_stated_uncalibrated() -> None:
    flat = _flat(_text(_GUIDE))
    assert "**These defaults are uncalibrated**" in flat
    assert "``0.45``" in flat and "``0.08``" in flat


# -- the other pages -------------------------------------------------------------


def test_index_lists_the_guide_and_caption_model() -> None:
    text = _text(_INDEX)
    toctree = text[text.index(".. toctree::") :]
    assert re.search(r"^\s+picture-search$", toctree, re.M)
    assert ":doc:`picture-search`" in text
    assert "ollama pull qwen3-vl:4b" in text


def test_contract_defines_picture_search_unavailable() -> None:
    flat = _flat(_text(_CONTRACT))
    assert "``picture_search_unavailable``" in flat
    for reason in ("unreachable", "model", "auth", "config"):
        assert f"``{reason}``" in flat
    assert "``null``, or why picture search cannot answer" in flat


def test_ingestion_states_the_health_result_contract() -> None:
    flat = _flat(_text(_INGESTION))
    assert "``HealthResult(reachable, message, reason)``" in flat
    assert "A plain ``(bool, str)`` pair is still accepted" in flat
    assert "makes one billed health completion" in flat

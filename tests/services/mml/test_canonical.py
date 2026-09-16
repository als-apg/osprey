"""Tests for the canonical ``ao.json``/``ad.json`` writer and reader.

The canonical files are hashed into every emitted artifact's header, so the
cases pin byte-level determinism: sorted keys, two-space indent, a trailing
newline, raw UTF-8, ``null`` for blank slots, and a refusal for any non-finite
float the normaliser failed to stringify.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from osprey.services.mml.canonical import (
    AD_FILENAME,
    AO_FILENAME,
    read_canonical,
    sha256_of,
    write_canonical,
)
from osprey.services.mml.systems import EXPORTS_KEY, IMPORT_ORDER_KEY


def _ao() -> dict:
    return {
        IMPORT_ORDER_KEY: ["SR", "BR"],
        EXPORTS_KEY: {"SR": {"exporter": "1.0"}},
        "SR": {
            "BPM": {
                "Monitor": {"ChannelNames": ["SR:BPM1:X", None], "Units": "mm"},
                "DeviceList": [[1, 1], [1, 2]],
                "_description": "Beam position — µm grade",
            },
        },
        "BR": {"QF": {"Setpoint": {"ChannelNames": ["BR:QF:SP"], "Range": ["-Inf", "Inf"]}}},
    }


def _ad() -> dict:
    return {"SR": {"Machine": "Quokka", "SubMachine": "SR", "Energy": 1.9}}


class TestWrite:
    def test_returns_paths_in_out_dir(self, tmp_path: Path) -> None:
        """The two paths are ao.json and ad.json under the output directory."""
        ao_path, ad_path = write_canonical(_ao(), _ad(), tmp_path)
        assert ao_path == tmp_path / AO_FILENAME
        assert ad_path == tmp_path / AD_FILENAME
        assert (AO_FILENAME, AD_FILENAME) == ("ao.json", "ad.json")
        assert ao_path.is_file() and ad_path.is_file()

    def test_creates_missing_out_dir(self, tmp_path: Path) -> None:
        """A missing data/mml directory is created, parents included."""
        out = tmp_path / "data" / "mml"
        ao_path, _ = write_canonical(_ao(), _ad(), out)
        assert ao_path.parent == out

    def test_exact_serialisation(self, tmp_path: Path) -> None:
        """Bytes equal json.dumps(sort_keys, indent=2, raw UTF-8) plus one newline."""
        ao_path, ad_path = write_canonical(_ao(), _ad(), tmp_path)
        expected = json.dumps(_ao(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        assert ao_path.read_bytes() == expected.encode("utf-8")
        assert ad_path.read_text(encoding="utf-8").endswith("}\n")
        assert not ad_path.read_text(encoding="utf-8").endswith("\n\n")

    def test_keys_sorted_at_every_depth(self, tmp_path: Path) -> None:
        """Keys come out sorted however they were inserted; lists keep their order."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        text = ao_path.read_text(encoding="utf-8")
        assert text.index('"BR"') < text.index('"SR"') < text.index(f'"{EXPORTS_KEY}"')
        body = text[text.index('"BPM"') :]
        assert body.index('"DeviceList"') < body.index('"Monitor"') < body.index('"_description"')
        assert json.loads(text)[IMPORT_ORDER_KEY] == ["SR", "BR"]

    def test_unicode_written_raw(self, tmp_path: Path) -> None:
        """Non-ASCII text is written as UTF-8, never as \\u escapes."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        text = ao_path.read_text(encoding="utf-8")
        assert "µm" in text and "—" in text
        assert "\\u" not in text

    def test_none_slot_is_null(self, tmp_path: Path) -> None:
        """A blank slot serialises as null with its index preserved."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        assert '"SR:BPM1:X",\n' in ao_path.read_text(encoding="utf-8")
        assert json.loads(ao_path.read_text())["SR"]["BPM"]["Monitor"]["ChannelNames"] == [
            "SR:BPM1:X",
            None,
        ]

    def test_crlf_never_introduced(self, tmp_path: Path) -> None:
        """No newline translation: the file holds LF only."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        assert b"\r" not in ao_path.read_bytes()


class TestDeterminism:
    def test_double_write_byte_identical(self, tmp_path: Path) -> None:
        """Writing the same content twice leaves identical bytes."""
        ao_path, ad_path = write_canonical(_ao(), _ad(), tmp_path)
        first = (ao_path.read_bytes(), ad_path.read_bytes())
        write_canonical(_ao(), _ad(), tmp_path)
        assert (ao_path.read_bytes(), ad_path.read_bytes()) == first

    def test_insertion_order_does_not_matter(self, tmp_path: Path) -> None:
        """Dicts with the same items in a different insertion order write the same bytes."""
        a, b = tmp_path / "a", tmp_path / "b"
        ao = _ao()
        reordered = dict(reversed(list(ao.items())))
        write_canonical(ao, _ad(), a)
        write_canonical(reordered, _ad(), b)
        assert (a / AO_FILENAME).read_bytes() == (b / AO_FILENAME).read_bytes()

    def test_identical_rewrite_leaves_file_untouched(self, tmp_path: Path) -> None:
        """An identical rewrite does not replace the file, so its mtime stays."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        os.utime(ao_path, ns=(1_000_000_000, 1_000_000_000))
        inode = ao_path.stat().st_ino
        write_canonical(_ao(), _ad(), tmp_path)
        assert ao_path.stat().st_mtime_ns == 1_000_000_000
        assert ao_path.stat().st_ino == inode

    def test_changed_content_replaces_file(self, tmp_path: Path) -> None:
        """Different content does replace the file."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        ao = _ao()
        ao["SR"]["BPM"]["Monitor"]["Units"] = "um"
        write_canonical(ao, _ad(), tmp_path)
        assert json.loads(ao_path.read_text())["SR"]["BPM"]["Monitor"]["Units"] == "um"

    def test_no_temp_files_left(self, tmp_path: Path) -> None:
        """Only the two canonical files remain after writing twice."""
        write_canonical(_ao(), _ad(), tmp_path)
        ao = _ao()
        ao["BR"]["QF"]["Setpoint"]["Units"] = "A"
        write_canonical(ao, _ad(), tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == [AD_FILENAME, AO_FILENAME]


class TestNonFinite:
    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_planted_non_finite_in_ao_refused(self, tmp_path: Path, value: float) -> None:
        """A float the normaliser missed trips allow_nan=False."""
        ao = _ao()
        ao["BR"]["QF"]["Setpoint"]["Tolerance"] = value
        with pytest.raises(ValueError):
            write_canonical(ao, _ad(), tmp_path)

    def test_refusal_writes_nothing(self, tmp_path: Path) -> None:
        """A refused AD leaves neither file behind, and never a half pair."""
        ad = _ad()
        ad["SR"]["Circumference"] = math.inf
        with pytest.raises(ValueError):
            write_canonical(_ao(), ad, tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_refusal_keeps_previous_files(self, tmp_path: Path) -> None:
        """A refused rewrite leaves the earlier canonical pair byte-identical."""
        ao_path, ad_path = write_canonical(_ao(), _ad(), tmp_path)
        before = (ao_path.read_bytes(), ad_path.read_bytes())
        ao = _ao()
        ao["SR"]["BPM"]["Monitor"]["Tolerance"] = float("inf")
        with pytest.raises(ValueError):
            write_canonical(ao, _ad(), tmp_path)
        assert (ao_path.read_bytes(), ad_path.read_bytes()) == before

    def test_stringified_non_finite_passes(self, tmp_path: Path) -> None:
        """The normaliser's Inf/-Inf/NaN strings are ordinary strings."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        assert '"-Inf"' in ao_path.read_text(encoding="utf-8")


class TestRead:
    def test_round_trip(self, tmp_path: Path) -> None:
        """read_canonical returns what write_canonical was given."""
        write_canonical(_ao(), _ad(), tmp_path)
        assert read_canonical(tmp_path) == (_ao(), _ad())

    def test_empty_ad_round_trip(self, tmp_path: Path) -> None:
        """An import with no AD writes and reads back an empty object."""
        write_canonical(_ao(), {}, tmp_path)
        assert (tmp_path / AD_FILENAME).read_text(encoding="utf-8") == "{}\n"
        assert read_canonical(tmp_path) == (_ao(), {})

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A directory without ao.json raises FileNotFoundError naming it."""
        with pytest.raises(FileNotFoundError, match="ao.json"):
            read_canonical(tmp_path)

    def test_bare_non_finite_token_refused(self, tmp_path: Path) -> None:
        """A hand-edited bare NaN/Infinity token is refused on read."""
        write_canonical(_ao(), _ad(), tmp_path)
        (tmp_path / AD_FILENAME).write_text('{"SR": {"Energy": NaN}}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="ad.json"):
            read_canonical(tmp_path)

    def test_non_object_refused(self, tmp_path: Path) -> None:
        """A canonical file whose top level is not an object is refused."""
        write_canonical(_ao(), _ad(), tmp_path)
        (tmp_path / AO_FILENAME).write_text("[]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="ao.json"):
            read_canonical(tmp_path)


class TestSha256:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        """sha256_of is the lowercase hex digest of the file bytes."""
        ao_path, _ = write_canonical(_ao(), _ad(), tmp_path)
        assert sha256_of(ao_path) == hashlib.sha256(ao_path.read_bytes()).hexdigest()

    def test_stable_across_identical_writes(self, tmp_path: Path) -> None:
        """Identical content hashes the same across two output directories."""
        a, _ = write_canonical(_ao(), _ad(), tmp_path / "a")
        b, _ = write_canonical(_ao(), _ad(), tmp_path / "b")
        assert sha256_of(a) == sha256_of(b)
        assert len(sha256_of(a)) == 64

    def test_large_file_chunked(self, tmp_path: Path) -> None:
        """A file larger than one read chunk hashes correctly."""
        path = tmp_path / "big.bin"
        data = os.urandom(3 * 1024 * 1024 + 17)
        path.write_bytes(data)
        assert sha256_of(path) == hashlib.sha256(data).hexdigest()

"""Conformance suite for ``osprey.bridges.core.artifacts``.

This file consolidates five bridge-core test modules, one per public function.
Each origin is kept as its own class so the test names stay verbatim (several
names — ``test_empty_and_none_are_safe`` most notably — appear in more than one
origin and would otherwise shadow each other in a single module namespace).

Sections:

* ``TestFetchArtifact``  — worker byte fetch, size guard, and the served
  Content-Type that is authoritative over any prediction
* ``TestArtifactIds``    — run-status ``artifacts`` field -> id strings
* ``TestArtifactDescriptors`` — run-status ``artifacts`` field -> descriptors
* ``TestExtForMime``     — mime -> extension allowlist
* ``TestSafeLabel``      — CR/LF stripping + length bound
* ``TestNeverRaiseContract`` — new here: pins the "a fetch failure never
  escapes" guarantee the module's docstring promises
"""

import httpx
import pytest

from osprey.bridges.core.artifacts import (
    PNG_MAGIC,
    artifact_descriptors,
    artifact_ids,
    ext_for_mime,
    fetch_artifact,
    safe_label,
)
from osprey.bridges.core.config import CoreConfig

# ---------------------------------------------------------------------------
# fetch_artifact: worker byte fetch + size guard, and the two facts a consumer
# routes on — the bytes themselves (is_png) and the Content-Type the worker
# served them under. Neither is a prediction, so neither can disagree with what
# was actually delivered.
# ---------------------------------------------------------------------------

PNG = PNG_MAGIC + b"body"
PDF = b"%PDF-1.7\n...."
CSV = b"x,sin_x\n0,0\n"

CFG = CoreConfig(worker_url="http://work:9190", dispatch_worker_token="work-token")


def _http(payloads):
    """payloads: artifact_id -> bytes, or (bytes, content_type), or None (404)."""

    def handler(request):
        aid = str(request.url).rsplit("/", 1)[-1]
        payload = payloads.get(aid)
        if payload is None:
            return httpx.Response(404)
        assert request.headers["Authorization"] == "Bearer work-token"
        data, content_type = payload if isinstance(payload, tuple) else (payload, None)
        headers = {"Content-Type": content_type} if content_type else {}
        return httpx.Response(200, content=data, headers=headers)

    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=CFG.trust_env)


class TestFetchArtifact:
    def test_returns_the_bytes_and_the_served_content_type(self):
        fetched = fetch_artifact(_http({"a": (PNG, "image/png")}), CFG, "R1", "a")
        assert fetched.data == PNG
        assert fetched.content_type == "image/png"

    def test_a_non_image_payload_is_returned_not_dropped(self):
        # osprey #503: the byte route serving CSV where the descriptor predicted a
        # PNG must not make the artifact vanish — the caller re-routes it.
        fetched = fetch_artifact(_http({"a": (CSV, "text/csv")}), CFG, "R1", "a")
        assert fetched.data == CSV
        assert fetched.content_type == "text/csv"
        assert fetched.is_png is False

    def test_is_png_reads_the_magic_bytes_not_the_header(self):
        # A header claiming PNG over text bytes is exactly the #503 failure; the
        # magic bytes are the unforgeable half.
        assert fetch_artifact(_http({"a": (CSV, "image/png")}), CFG, "R1", "a").is_png is False
        assert fetch_artifact(_http({"a": (PNG, "text/csv")}), CFG, "R1", "a").is_png is True

    def test_content_type_parameters_are_stripped(self):
        # Starlette appends "; charset=utf-8" to any text/* media type.
        fetched = fetch_artifact(_http({"a": (CSV, "text/csv; charset=utf-8")}), CFG, "R1", "a")
        assert fetched.content_type == "text/csv"

    def test_a_missing_content_type_is_none_not_a_guess(self):
        assert fetch_artifact(_http({"a": PDF}), CFG, "R1", "a").content_type is None

    def test_custom_max_bytes_enforced(self):
        big = PNG_MAGIC + b"x" * 100
        assert fetch_artifact(_http({"a": big}), CFG, "R1", "a", max_bytes=10) is None
        assert fetch_artifact(_http({"a": big}), CFG, "R1", "a", max_bytes=1000).data == big

    def test_404_returns_none(self):
        assert fetch_artifact(_http({}), CFG, "R1", "missing") is None


# ---------------------------------------------------------------------------
# artifact_ids: normalize the run-status ``artifacts`` field to id strings.
#
# Covers the three inputs the deploy window can produce — osprey #363 descriptor
# dicts, bare id strings from an older worker, and a mix — plus the malformed
# shapes that must be skipped rather than raised on.
# ---------------------------------------------------------------------------


class TestArtifactIds:
    @staticmethod
    def _descriptor(aid):
        """A realistic #363 descriptor dict."""
        return {
            "artifact_id": aid,
            "filename": f"{aid}.png",
            "source_mime": "image/png",
            "delivered_mime": "image/png",
            "convertible": True,
        }

    def test_descriptor_dicts_yield_ids(self):
        artifacts = [self._descriptor("art-1"), self._descriptor("art-2")]
        assert artifact_ids(artifacts) == ["art-1", "art-2"]

    def test_bare_strings_pass_through_unchanged(self):
        # Back-compat: an older worker still answering with id strings mid-deploy.
        assert artifact_ids(["art-1", "art-2"]) == ["art-1", "art-2"]

    def test_mixed_dicts_and_strings(self):
        assert artifact_ids([self._descriptor("art-1"), "art-2"]) == ["art-1", "art-2"]

    def test_empty_and_none_are_safe(self):
        assert artifact_ids([]) == []
        assert artifact_ids(None) == []

    def test_malformed_entries_are_skipped_not_raised(self):
        artifacts = [
            self._descriptor("keep-1"),
            {"filename": "no-id.png"},  # dict without artifact_id
            {"artifact_id": None},  # null id
            {"artifact_id": ""},  # empty id
            {"artifact_id": 123},  # non-str id (must not leak an int into a URL)
            "",  # empty string
            123,  # wrong type entirely
            None,  # bare None element
            self._descriptor("keep-2"),
        ]
        assert artifact_ids(artifacts) == ["keep-1", "keep-2"]


# ---------------------------------------------------------------------------
# artifact_descriptors: normalize the run-status ``artifacts`` field to
# descriptor dicts, keeping every hint (filename, delivered_mime) a delivery
# path can use to NAME an artifact. It deliberately does not route: what an
# artifact turns out to be is only knowable once its bytes have been served.
#
# Covers the same input shapes as ``artifact_ids`` — osprey #363 descriptor
# dicts, bare id strings from an older worker, a mix, malformed shapes to skip.
# ---------------------------------------------------------------------------


class TestArtifactDescriptors:
    @staticmethod
    def _descriptor(aid, mime="image/png", **overrides):
        """A realistic #363 descriptor dict."""
        d = {
            "artifact_id": aid,
            "filename": f"{aid}.bin",
            "source_mime": mime,
            "delivered_mime": mime,
            "convertible": True,
        }
        d.update(overrides)
        return d

    def test_descriptor_dicts_pass_through_whole(self):
        desc = self._descriptor("art-1")
        assert artifact_descriptors([desc]) == [desc]

    def test_a_bare_string_becomes_a_descriptor_with_no_hints(self):
        # Back-compat: an older worker still answering with id strings mid-deploy.
        assert artifact_descriptors(["art-1"]) == [{"artifact_id": "art-1"}]

    def test_every_mime_is_kept_in_one_list(self):
        # No mime-based routing here: a PNG prediction and a markdown prediction
        # are the same kind of hint, and either can be wrong.
        png = self._descriptor("keep-png", "image/png")
        md = self._descriptor("keep-md", "text/markdown")
        assert artifact_descriptors([png, md]) == [png, md]

    def test_a_descriptor_without_a_mime_is_kept(self):
        desc = {"artifact_id": "art-5", "filename": "art-5.bin"}
        assert artifact_descriptors([desc]) == [desc]

    def test_malformed_entries_are_skipped_not_raised_on(self):
        artifacts = [
            {"filename": "no-id.png"},  # dict without artifact_id
            {"artifact_id": None},  # null id
            {"artifact_id": ""},  # empty id
            {"artifact_id": 123},  # non-str id (must not leak an int into a URL)
            "",  # empty string
            123,  # wrong type entirely
            None,  # bare None element
        ]
        assert artifact_descriptors(artifacts) == []

    def test_a_malformed_entry_does_not_cost_its_valid_siblings(self):
        keep = self._descriptor("keep")
        assert artifact_descriptors([{"filename": "no-id.png"}, keep, None, "bare"]) == [
            keep,
            {"artifact_id": "bare"},
        ]

    def test_empty_and_none_are_safe(self):
        assert artifact_descriptors([]) == []
        assert artifact_descriptors(None) == []


# ---------------------------------------------------------------------------
# ext_for_mime: map a ``delivered_mime`` to a file extension via a fixed
# allowlist, falling back to ``.bin`` for anything not in it.
# ---------------------------------------------------------------------------

MAPPED = [
    ("text/html", ".html"),
    ("text/markdown", ".md"),
    ("text/plain", ".txt"),
    ("application/pdf", ".pdf"),
    ("text/csv", ".csv"),
    ("application/json", ".json"),
    ("image/jpeg", ".jpg"),
    ("image/svg+xml", ".svg"),
]


class TestExtForMime:
    @pytest.mark.parametrize("mime,ext", MAPPED)
    def test_mapped_mimes(self, mime, ext):
        assert ext_for_mime(mime) == ext

    def test_unknown_mime_falls_back_to_bin(self):
        assert ext_for_mime("application/x-tar") == ".bin"

    def test_none_falls_back_to_bin(self):
        assert ext_for_mime(None) == ".bin"

    def test_empty_string_falls_back_to_bin(self):
        assert ext_for_mime("") == ".bin"


# ---------------------------------------------------------------------------
# safe_label: strip CR/LF and bound the length of a worker-supplied name.
#
# Shared by a channel's user-visible label and an attachment filename.
# ---------------------------------------------------------------------------


class TestSafeLabel:
    def test_plain_name_passes_through(self):
        assert safe_label("plot.png", "fallback") == "plot.png"

    def test_embedded_crlf_is_stripped(self):
        assert safe_label("plot\r\n.png", "fallback") == "plot.png"

    def test_surrounding_whitespace_is_trimmed(self):
        assert safe_label("  plot.png  ", "fallback") == "plot.png"

    def test_empty_name_falls_back(self):
        assert safe_label("", "fallback") == "fallback"

    def test_whitespace_only_name_falls_back(self):
        assert safe_label("   ", "fallback") == "fallback"

    def test_none_name_falls_back(self):
        assert safe_label(None, "fallback") == "fallback"

    def test_long_name_is_bounded_to_200_chars(self):
        long_name = "a" * 300
        result = safe_label(long_name, "fallback")
        assert len(result) == 200
        assert result == "a" * 200

    def test_name_that_becomes_empty_after_stripping_falls_back(self):
        # All CR/LF, nothing left after cleaning -> fallback, then bounded.
        result = safe_label("\r\n\r\n", "fallback")
        assert result == "fallback"

    def test_fallback_itself_is_bounded(self):
        long_fallback = "b" * 300
        result = safe_label("", long_fallback)
        assert len(result) == 200
        assert result == "b" * 200


# ---------------------------------------------------------------------------
# NEW (not in the ported suite): the never-raise contract.
#
# The module promises that a failed fetch degrades delivery to text-only rather
# than losing the answer, so no exception may escape fetch_artifact — including
# the URL-construction failures that do NOT subclass httpx.HTTPError.
# ---------------------------------------------------------------------------


class TestNeverRaiseContract:
    def test_control_char_in_artifact_id_returns_none(self):
        # httpx.InvalidURL is raised at URL-construction time and is not an
        # httpx.HTTPError, so only a broad except keeps it from escaping.
        assert fetch_artifact(_http({}), CFG, "R1", "bad\nid") is None

    def test_transport_exception_returns_none(self):
        def boom(request):
            raise httpx.ConnectError("worker unreachable", request=request)

        http = httpx.Client(transport=httpx.MockTransport(boom), trust_env=CFG.trust_env)
        assert fetch_artifact(http, CFG, "R1", "a") is None

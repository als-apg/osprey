"""Conformance suite for ``osprey.bridges.core.artifacts``.

This file consolidates five bridge-core test modules, one per public function.
Each origin is kept as its own class so the test names stay verbatim (several
names — ``test_empty_and_none_are_safe`` most notably — appear in more than one
origin and would otherwise shadow each other in a single module namespace).

Sections:

* ``TestFetchArtifact``  — worker byte fetch + size/PNG guards
* ``TestArtifactIds``    — run-status ``artifacts`` field -> id strings
* ``TestSplitArtifacts`` — mime-routed split into image ids / doc descriptors
* ``TestExtForMime``     — mime -> extension allowlist
* ``TestSafeLabel``      — CR/LF stripping + length bound
* ``TestNeverRaiseContract`` — new here: pins the "a fetch failure never
  escapes" guarantee the module's docstring promises
"""

import httpx
import pytest

from osprey.bridges.core.artifacts import (
    PNG_MAGIC,
    artifact_ids,
    ext_for_mime,
    fetch_artifact,
    safe_label,
    split_artifacts,
)
from osprey.bridges.core.config import CoreConfig

# ---------------------------------------------------------------------------
# fetch_artifact: worker byte fetch + size/PNG guards, incl. the non-PNG path
# a document-delivering channel relies on (require_png=False) and a custom
# max_bytes.
# ---------------------------------------------------------------------------

PNG = PNG_MAGIC + b"body"
PDF = b"%PDF-1.7\n...."

CFG = CoreConfig(worker_url="http://work:9190", dispatch_worker_token="work-token")


def _http(payloads):
    """payloads: artifact_id -> bytes (200) or None (404)."""

    def handler(request):
        aid = str(request.url).rsplit("/", 1)[-1]
        data = payloads.get(aid)
        if data is None:
            return httpx.Response(404)
        assert request.headers["Authorization"] == "Bearer work-token"
        return httpx.Response(200, content=data)

    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=CFG.trust_env)


class TestFetchArtifact:
    def test_png_default_accepts_png(self):
        assert fetch_artifact(_http({"a": PNG}), CFG, "R1", "a") == PNG

    def test_png_default_rejects_non_png(self):
        assert fetch_artifact(_http({"a": PDF}), CFG, "R1", "a") is None

    def test_require_png_false_accepts_non_png(self):
        # document path: attach delivered_mime bytes verbatim, no PNG requirement.
        assert fetch_artifact(_http({"a": PDF}), CFG, "R1", "a", require_png=False) == PDF

    def test_custom_max_bytes_enforced(self):
        big = PNG_MAGIC + b"x" * 100
        assert fetch_artifact(_http({"a": big}), CFG, "R1", "a", max_bytes=10) is None
        assert fetch_artifact(_http({"a": big}), CFG, "R1", "a", max_bytes=1000) == big

    def test_404_returns_none(self):
        assert fetch_artifact(_http({}), CFG, "R1", "missing") is None

    def test_size_guard_applies_even_when_png_not_required(self):
        assert (
            fetch_artifact(_http({"a": PDF}), CFG, "R1", "a", require_png=False, max_bytes=2)
            is None
        )


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
# split_artifacts: route the run-status ``artifacts`` field into an image-id
# bucket (PNG, inline-image path) and a doc-descriptor bucket (everything else).
#
# Covers the same input shapes as ``artifact_ids`` (osprey #363 descriptor
# dicts, bare id strings from an older worker, a mix, malformed shapes to skip)
# plus the mime-based routing split_artifacts adds on top.
# ---------------------------------------------------------------------------


class TestSplitArtifacts:
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

    def test_png_descriptor_goes_to_image_bucket_by_id(self):
        image_ids, doc_descriptors = split_artifacts([self._descriptor("art-1", "image/png")])
        assert image_ids == ["art-1"]
        assert doc_descriptors == []

    def test_bare_string_goes_to_image_bucket(self):
        # Back-compat: an older worker with no mime at all -> existing PNG path.
        image_ids, doc_descriptors = split_artifacts(["art-1"])
        assert image_ids == ["art-1"]
        assert doc_descriptors == []

    def test_image_jpeg_goes_to_doc_bucket_as_whole_dict(self):
        desc = self._descriptor("art-2", "image/jpeg")
        image_ids, doc_descriptors = split_artifacts([desc])
        assert image_ids == []
        assert doc_descriptors == [desc]

    def test_image_svg_goes_to_doc_bucket(self):
        desc = self._descriptor("art-3", "image/svg+xml")
        image_ids, doc_descriptors = split_artifacts([desc])
        assert image_ids == []
        assert doc_descriptors == [desc]

    def test_document_mime_goes_to_doc_bucket(self):
        desc = self._descriptor("art-4", "text/markdown")
        image_ids, doc_descriptors = split_artifacts([desc])
        assert image_ids == []
        assert doc_descriptors == [desc]

    def test_missing_delivered_mime_key_goes_to_doc_bucket(self):
        desc = {"artifact_id": "art-5", "filename": "art-5.bin"}
        assert "delivered_mime" not in desc
        image_ids, doc_descriptors = split_artifacts([desc])
        assert image_ids == []
        assert doc_descriptors == [desc]

    def test_malformed_entries_are_skipped_from_both_buckets(self):
        artifacts = [
            {"filename": "no-id.png"},  # dict without artifact_id
            {"artifact_id": None},  # null id
            {"artifact_id": ""},  # empty id
            {"artifact_id": 123},  # non-str id
            "",  # empty string
            123,  # wrong type entirely
            None,  # bare None element
        ]
        image_ids, doc_descriptors = split_artifacts(artifacts)
        assert image_ids == []
        assert doc_descriptors == []

    def test_mixed_input_keeps_each_id_in_exactly_one_bucket(self):
        png = self._descriptor("keep-png", "image/png")
        jpeg = self._descriptor("keep-jpeg", "image/jpeg")
        md = self._descriptor("keep-md", "text/markdown")
        no_mime = {"artifact_id": "keep-no-mime", "filename": "x.bin"}
        bare = "keep-bare"
        artifacts = [
            png,
            {"filename": "no-id.png"},
            jpeg,
            "",
            md,
            None,
            no_mime,
            bare,
            {"artifact_id": None},
        ]
        image_ids, doc_descriptors = split_artifacts(artifacts)

        all_doc_ids = {d["artifact_id"] for d in doc_descriptors}
        assert set(image_ids) & all_doc_ids == set()

        assert image_ids == ["keep-png", "keep-bare"]
        assert doc_descriptors == [jpeg, md, no_mime]

    def test_empty_and_none_are_safe(self):
        assert split_artifacts([]) == ([], [])
        assert split_artifacts(None) == ([], [])


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

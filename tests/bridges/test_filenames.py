"""The collision-free naming pass a bridge runs over one delivery's files.

Pure functions, no transport: a stem per artifact settled before any fetch, then an
extension from the served type. Every name stays one safe path segment, and two
artifacts of one delivery never reach one name.
"""

import pytest

from osprey.bridges.core import unique_stems, upload_name, upload_stem

# ==========================================================================
# Upload filenames stay inside the artifact tree
# ==========================================================================


@pytest.mark.parametrize(
    ("label", "extension", "expected"),
    [
        ("plot", ".png", "plot.png"),
        ("plot.png", ".png", "plot.png"),  # not doubled
        ("PLOT.PNG", ".png", "PLOT.PNG"),  # case-insensitive match
        # A predicted name whose extension the delivery contradicted: the mime
        # wins and the stale extension is replaced, never stacked.
        ("data.png", ".csv", "data.csv"),
        ("run_2026.04.01_orbit", ".csv", "run_2026.04.01_orbit.csv"),  # not an extension
        ("orbit report", ".pdf", "orbit_report.pdf"),
        # Separators become "_" and the leading dot/underscore run is stripped, so a
        # traversal attempt collapses into one ordinary filename.
        ("../../etc/passwd", ".bin", "etc_passwd.bin"),
        ("with\nnewline", ".txt", "withnewline.txt"),
        ("", ".png", "fallback.png"),
        (None, ".png", "fallback.png"),
        (42, ".png", "fallback.png"),
        ("..", ".png", "fallback.png"),
        ("/", ".png", "fallback.png"),
        ("x" * 300, ".png", "x" * 120 + ".png"),
    ],
)
def test_upload_names_are_a_single_safe_segment(label, extension, expected):
    name = upload_name(upload_stem(label, "fallback"), extension)
    assert name == expected
    assert "/" not in name
    assert name not in (".", "..")


def test_upload_name_falls_back_twice_when_even_the_fallback_is_unusable():
    assert upload_name(upload_stem(None, ".."), ".png") == "artifact.png"


# ==========================================================================
# Two artifacts never claim one upload name
# ==========================================================================


def test_same_named_artifacts_are_kept_apart():
    # The worker names an artifact by basename, so a run whose steps each wrote their
    # own plot.png sends two descriptors with one filename. They share a DAV directory,
    # so an undisambiguated second upload would overwrite the first.
    stems = unique_stems(
        [
            {"artifact_id": "a1", "filename": "plot.png"},
            {"artifact_id": "a2", "filename": "plot.png"},
        ]
    )
    assert stems["a1"] == "plot.png"
    # The id slice goes AHEAD of the trailing dot-suffix, so the name still reads as one.
    assert stems["a2"] == "plot-a2.png"


def test_stems_that_differ_but_name_one_file_are_kept_apart():
    # The same gap at the unit level. This particular pair needs a cross-bucket route to
    # occur for real (a doc-bucket artifact whose bytes serve as image/png, next to an
    # image-bucket plot.png), so it pins the MECHANISM rather than a worker output shape.
    stems = unique_stems(
        [
            {"artifact_id": "a1", "filename": "plot.png"},
            {"artifact_id": "a2", "filename": "plot"},
        ]
    )
    names = [upload_name(stems[aid], ".png") for aid in ("a1", "a2")]
    assert names == ["plot.png", "plot-a2.png"]


def test_a_disambiguated_truncated_stem_is_still_one_safe_segment():
    # _segment strips its dot runs BEFORE truncating at 120, so a long name cut exactly
    # at a dot ends in one and _disambiguate's partition yields an empty tail. The
    # trailing dot is cosmetic; what must hold is that the name is still a single
    # segment dav_mkcol_put accepts.
    long_name = "a" * 119 + "." + "b" * 30
    stems = unique_stems(
        [{"artifact_id": "a1", "filename": long_name}, {"artifact_id": "a2", "filename": long_name}]
    )
    for stem in stems.values():
        name = upload_name(stem, ".png")
        assert "/" not in name
        assert name not in (".", "..")
        assert name.strip(".")
    assert len(set(stems.values())) == 2


@pytest.mark.parametrize(
    ("first", "second", "extension"),
    [
        # Only a KNOWN extension may be stripped when keying: treating ".v2" as one
        # keys these apart, and they then collide on report.v2.pdf.
        ("report.v2", "report.v2.pdf", ".pdf"),
        # One strip is not enough — plot.png.pdf must lose both before the comparison.
        ("plot.png", "plot.png.pdf", ".pdf"),
        # upload_name's already-suffixed check is case-insensitive, so the key must
        # fold case too.
        ("a.PDF", "a", ".pdf"),
        ("report", "report.html", ".html"),
    ],
)
def test_names_that_would_collide_after_the_extension_are_kept_apart(first, second, extension):
    stems = unique_stems(
        [{"artifact_id": "d1", "filename": first}, {"artifact_id": "d2", "filename": second}]
    )
    names = [upload_name(stems[aid], extension) for aid in ("d1", "d2")]
    assert names[0] != names[1], f"{first!r} and {second!r} both upload as {names[0]!r}"


def test_suffixes_that_are_not_extensions_are_not_disambiguated():
    # The flip side of stripping only KNOWN extensions: these two never collide, so
    # neither should be renamed.
    stems = unique_stems(
        [
            {"artifact_id": "d1", "filename": "report.draft"},
            {"artifact_id": "d2", "filename": "report.final"},
        ]
    )
    assert [upload_name(stems[aid], ".pdf") for aid in ("d1", "d2")] == [
        "report.draft.pdf",
        "report.final.pdf",
    ]


def test_stems_that_differ_only_by_case_are_kept_apart():
    # A case-insensitive DAV backend would treat these as one path.
    stems = unique_stems(
        [{"artifact_id": "a1", "filename": "PLOT"}, {"artifact_id": "a2", "filename": "plot"}]
    )
    assert len({stem.lower() for stem in stems.values()}) == 2


def test_stems_stay_unique_when_even_the_id_slice_collides():
    # Distinct ids can sanitize to one slice, so the slice alone is not enough.
    stems = unique_stems(
        [
            {"artifact_id": "step/one", "filename": "plot"},
            {"artifact_id": "step:one", "filename": "plot"},
            {"artifact_id": "step one", "filename": "plot"},
        ]
    )
    assert sorted(stems.values()) == ["plot", "plot-step_one", "plot-step_one-2"]


def test_stems_are_assigned_across_both_buckets():
    # Images and documents land in the same directory, so a collision between the two
    # buckets is the same overwrite as one within either.
    #
    # Both descriptors are shapes the worker can actually emit: predict_delivery only
    # ever reports a passthrough source mime or image/png, and _predicted_filename
    # forces the .png suffix on the latter — so an image descriptor always carries a
    # .png name, and application/pdf passes through keeping its own.
    stems = unique_stems(
        [
            {"artifact_id": "img", "delivered_mime": "image/png", "filename": "report.png"},
            {"artifact_id": "doc", "delivered_mime": "application/pdf", "filename": "report.pdf"},
        ]
    )
    assert len(set(stems.values())) == 2

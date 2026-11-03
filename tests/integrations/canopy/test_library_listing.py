"""The listing of a CANOPY screen library, and the check that its report still covers it.

``fixtures/listing-vectors.json`` is shared byte for byte with CANOPY's
implementation of the same definition (``canopy_mcp.listing``). Its cases are
entry lists run through the pure ``judge`` only: a case-insensitive host cannot
hold two names that differ only in case, so no vector touches a filesystem.

``library_is_fresh`` reads three things from a report — ``generator.schema``,
``files.digest`` and ``files.list`` — so the reports below are written by hand
with exactly those. Nothing here needs CANOPY installed.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

import osprey
from osprey.integrations.canopy import library

VECTORS_PATH = Path(__file__).parent / "fixtures" / "listing-vectors.json"
VECTORS_BYTES = VECTORS_PATH.read_bytes()
VECTORS = json.loads(VECTORS_BYTES)
CASES = {case["name"]: case for case in VECTORS["cases"]}

#: The list the image context's ``.dockerignore`` is derived from.
DOCKERIGNORE = Path(osprey.__file__).parent / "templates" / "project" / "dockerignore"

SCHEMA = "canopy-library-report/1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plain_listing(root: Path) -> bytes:
    """The listing line format, computed without the judge: every regular file, sorted."""
    screens = root / "screens"
    rows: list[tuple[bytes, str]] = []
    for dirpath, _dirnames, filenames in os.walk(screens):
        for name in filenames:
            path = Path(dirpath) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                continue
            rel = path.relative_to(screens).as_posix()
            rows.append((rel.encode("utf-8"), f"{_sha(path.read_bytes())}  screens/{rel}\n"))
    return "".join(line for _key, line in sorted(rows)).encode("utf-8")


# --- the module ------------------------------------------------------------------------


def test_the_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(Path(library.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add((node.module or "").split(".")[0])
    assert imported <= sys.stdlib_module_names, sorted(imported - sys.stdlib_module_names)


# --- the vector file -------------------------------------------------------------------


def test_the_vector_file_is_ascii_only() -> None:
    assert VECTORS_BYTES.isascii()


def test_the_vector_file_has_no_trailing_whitespace() -> None:
    assert b"\r" not in VECTORS_BYTES and b"\t" not in VECTORS_BYTES
    lines = VECTORS_BYTES.split(b"\n")
    assert [n for n, line in enumerate(lines, 1) if line != line.rstrip()] == []


def test_the_vector_file_ends_in_exactly_one_newline() -> None:
    assert VECTORS_BYTES.endswith(b"\n") and not VECTORS_BYTES.endswith(b"\n\n")


def test_the_vector_file_holds_the_cases_the_contract_names() -> None:
    paths = {entry["path"] for case in VECTORS["cases"] for entry in case["entries"]}
    assert {
        ".DS_Store",
        "screens/.DS_Store",
        "README.md",
        "screens/prefetched-models/a.screen.json",
        "screens/img.log/a.png",
        "screens/Common/a.screen.json",
        "screens/common/b.screen.json",
        "screens/a/Top.screen.json",
        "screens/a/top.screen.json",
    } <= paths
    codes = {refusal["code"] for case in VECTORS["cases"] for refusal in case["expect"]["refusals"]}
    assert codes == set(library._REASONS)


# --- judge -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_judge_matches_the_vectors(name: str) -> None:
    case = CASES[name]
    judged = library.judge(case["entries"])
    assert judged.list_bytes.decode("utf-8") == case["expect"]["list"]
    assert judged.digest == case["expect"]["digest"]
    assert list(judged.refusals) == case["expect"]["refusals"]


@pytest.mark.parametrize("name", list(CASES))
def test_judge_does_not_depend_on_the_order_of_entries(name: str) -> None:
    case = CASES[name]
    judged = library.judge(list(reversed(case["entries"])))
    assert judged.list_bytes.decode("utf-8") == case["expect"]["list"]
    assert list(judged.refusals) == case["expect"]["refusals"]


@pytest.mark.parametrize("name", list(CASES))
def test_the_digest_is_the_sha256_of_the_list(name: str) -> None:
    expect = CASES[name]["expect"]
    assert expect["digest"] == "sha256:" + _sha(expect["list"].encode("utf-8"))
    assert library.library_digest(expect["list"].encode("utf-8")) == expect["digest"]


def test_a_list_line_is_the_hash_two_spaces_and_the_path() -> None:
    entries = [
        {"path": "screens/b c.png", "kind": "file", "mode": 0o644, "sha256": "2" * 64},
        {"path": "screens/a.screen.json", "kind": "file", "mode": 0o644, "sha256": "1" * 64},
    ]
    judged = library.judge(entries)
    assert (
        judged.list_bytes
        == f"{'1' * 64}  screens/a.screen.json\n{'2' * 64}  screens/b c.png\n".encode()
    )
    assert judged.files == (("screens/a.screen.json", "1" * 64), ("screens/b c.png", "2" * 64))
    assert judged.paths == {"screens/a.screen.json", "screens/b c.png"}


def test_a_ds_store_file_is_neither_listed_nor_refused() -> None:
    expect = CASES["a regular file named .DS_Store is skipped"]["expect"]
    assert expect["refusals"] == [] and ".DS_Store" not in expect["list"]


def test_a_colliding_pair_names_both_spellings() -> None:
    expect = CASES["a case-only directory pair"]["expect"]
    assert expect["refusals"] == [
        {"path": "screens/Common", "code": "case_collision"},
        {"path": "screens/common", "code": "case_collision"},
    ]


def test_a_case_only_file_pair_is_refused() -> None:
    expect = CASES["a case-only file pair"]["expect"]
    assert expect["refusals"] == [
        {"path": "screens/a/Top.screen.json", "code": "case_collision"},
        {"path": "screens/a/top.screen.json", "code": "case_collision"},
    ]


@pytest.mark.parametrize("name", VECTORS["ignoredNames"]["refused"])
def test_ignored_names_are_refused(name: str) -> None:
    assert library.is_ignored_name(name)


@pytest.mark.parametrize("name", VECTORS["ignoredNames"]["accepted"])
def test_other_names_are_not_ignored(name: str) -> None:
    assert not library.is_ignored_name(name)


def test_refusal_text_is_one_ascii_line_that_names_the_file() -> None:
    for case in VECTORS["cases"]:
        for refusal in case["expect"]["refusals"]:
            text = library.refusal_text(refusal)
            assert text.isascii() and "\n" not in text and "\r" not in text
            assert text.startswith(f"{refusal['code']}: ")
            assert "delete" in text
    assert library.refusal_text({"path": "screens/a\nb.png", "code": "bad_name"}).startswith(
        "bad_name: screens/a\\nb.png: "
    )


# --- the project dockerignore ----------------------------------------------------------


def _dockerignore_patterns() -> list[str]:
    """Every non-negated pattern of the project ``dockerignore``, as written."""
    lines = (line.strip() for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith(("#", "!"))]


def _segment_names(segment: str) -> list[str]:
    """Names one pattern segment matches: ``*`` as nothing and as text, a class as each member."""
    assert not set(segment) & set("?{}\\!"), f"a glob form this test cannot spell out: {segment}"
    choices = []
    for token in re.findall(r"\[[^\]]*\]|\*|[^\[*]+", segment):
        if token == "*":
            choices.append(["", "x"])
        elif token.startswith("["):
            choices.append(list(token[1:-1]))
        else:
            choices.append([token])
    names = ["".join(parts) for parts in itertools.product(*choices)]
    return [name for name in names if name]


def _pattern_paths(pattern: str) -> list[list[str]]:
    """The paths, as segment lists, that ``pattern`` matches below any directory."""
    segments = pattern.removeprefix("**/").strip("/").split("/")
    assert "**" not in segments
    return [list(path) for path in itertools.product(*map(_segment_names, segments))]


DOCKERIGNORE_PATTERNS = _dockerignore_patterns()


def test_the_dockerignore_still_names_what_the_listing_was_written_against() -> None:
    assert {"Dockerfile", "*.log", "var/", "**/prefetched-models/", ".env*", "*.py[cod]"} <= set(
        DOCKERIGNORE_PATTERNS
    )


@pytest.mark.parametrize("pattern", DOCKERIGNORE_PATTERNS)
def test_every_dockerignored_name_is_refused_by_the_name_predicate(pattern: str) -> None:
    """The image context matches each pattern at any depth, so below ``screens/`` too.

    A path the context prunes must not be on the list the report is computed
    over: for every path a pattern matches, the listing's name predicate has to
    refuse one of its segments.
    """
    paths = _pattern_paths(pattern)
    assert paths
    for segments in paths:
        assert any(library.is_ignored_name(segment) for segment in segments), segments


@pytest.mark.parametrize("pattern", DOCKERIGNORE_PATTERNS)
def test_nothing_a_dockerignore_pattern_matches_is_ever_listed(pattern: str) -> None:
    for segments in _pattern_paths(pattern):
        matched = "/".join(["screens", "a", *segments])
        as_file = {"path": matched, "kind": "file", "mode": 0o644, "sha256": "0" * 64}
        inside = {**as_file, "path": f"{matched}/x.png"}
        assert library.judge([as_file]).files == ()
        judged = library.judge([inside])
        assert judged.files == ()
        assert len(judged.refusals) == 1
        assert f"{matched}/".startswith(f"{judged.refusals[0]['path']}/")


# --- scan ------------------------------------------------------------------------------


def _tree(root: Path) -> None:
    (root / "screens" / "a").mkdir(parents=True)
    (root / "screens" / "a" / "top.screen.json").write_bytes(b"{}\n")
    (root / "screens" / "logo.png").write_bytes(b"png")
    for path in (root, root / "screens", root / "screens" / "a"):
        path.chmod(0o755)
    (root / "screens" / "a" / "top.screen.json").chmod(0o644)
    (root / "screens" / "logo.png").chmod(0o644)


def test_scan_reports_every_entry_with_kind_mode_and_hash(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "screens" / "a" / "top.screen.json").chmod(0o640)
    (tmp_path / "LIBRARY-REPORT.json").write_bytes(b"{}\n")
    (tmp_path / "screens" / "link.png").symlink_to("logo.png")
    (tmp_path / "screens" / "dirlink").symlink_to("a", target_is_directory=True)
    os.mkfifo(tmp_path / "screens" / "pipe")
    (tmp_path / "extra" / "deep").mkdir(parents=True)
    (tmp_path / "extra" / "deep" / "x.png").write_bytes(b"x")

    entries = library.scan(tmp_path)
    by_path = {entry["path"]: entry for entry in entries}
    assert {path: entry["kind"] for path, entry in by_path.items()} == {
        "LIBRARY-REPORT.json": "file",
        "extra": "dir",
        "screens": "dir",
        "screens/a": "dir",
        "screens/a/top.screen.json": "file",
        "screens/dirlink": "symlink",
        "screens/link.png": "symlink",
        "screens/logo.png": "file",
        "screens/pipe": "other",
    }
    assert [entry["path"] for entry in entries] == list(by_path)
    assert by_path["screens/a/top.screen.json"] == {
        "path": "screens/a/top.screen.json",
        "kind": "file",
        "mode": 0o640,
        "sha256": _sha(b"{}\n"),
    }
    assert [path for path, entry in by_path.items() if "sha256" in entry] == [
        "LIBRARY-REPORT.json",
        "screens/a/top.screen.json",
        "screens/logo.png",
    ]


def test_scan_of_a_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        library.scan(tmp_path / "absent")


def test_library_list_keeps_the_line_format_on_a_clean_tree(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "screens" / "a" / "b c").mkdir()
    (tmp_path / "screens" / "a" / "b c" / "Z.PNG").write_bytes(b"z")
    (tmp_path / "LIBRARY-REPORT.json").write_bytes(b"{}\n")
    judged = library.judge(library.scan(tmp_path))
    assert judged.refusals == ()
    assert library.library_list(tmp_path) == judged.list_bytes == _plain_listing(tmp_path)
    assert library.library_digest(library.library_list(tmp_path)) == judged.digest
    assert judged.digest == "sha256:" + _sha(_plain_listing(tmp_path))


def test_library_list_leaves_out_what_the_listing_refuses(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "screens" / "notes.txt").write_bytes(b"n")
    (tmp_path / "screens" / ".DS_Store").write_bytes(b"d")
    listed = library.library_list(tmp_path).decode("utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in listed] == [
        "screens/a/top.screen.json",
        "screens/logo.png",
    ]


def test_library_list_of_a_missing_library_is_empty(tmp_path: Path) -> None:
    assert library.library_list(tmp_path / "absent") == b""


# --- freshness -------------------------------------------------------------------------


def write_report(root: Path, *, schema: str = SCHEMA) -> dict:
    """A report for ``root`` as it stands, holding what the freshness check reads."""
    judged = library.judge(library.scan(root))
    report = {
        "generator": {"schema": schema, "canopy_version": "0.0.0"},
        "files": {
            "digest": judged.digest,
            "list": [{"path": path, "sha256": sha} for path, sha in judged.files],
        },
    }
    _write_json(root, report)
    (root / "LIBRARY-REPORT.md").write_bytes(b"# report\n")
    (root / "LIBRARY-REPORT.md").chmod(0o644)
    return report


def _write_json(root: Path, report: object) -> None:
    (root / "LIBRARY-REPORT.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    (root / "LIBRARY-REPORT.json").chmod(0o644)


@pytest.fixture
def fresh(tmp_path: Path) -> Path:
    """A library of two files with a report written for exactly those."""
    root = tmp_path / "library"
    root.mkdir()
    _tree(root)
    write_report(root)
    return root


def _refused(root: Path) -> list[str]:
    verdict = library.library_is_fresh(root)
    assert not verdict and verdict.fresh is False
    assert all(line.isascii() and "\n" not in line for line in verdict.lines)
    return list(verdict.lines)


def _regeneration(root: Path) -> str:
    return f"python -m canopy_mcp.report {root} --served-at library --write"


def test_a_library_with_a_current_report_is_fresh(fresh: Path) -> None:
    verdict = library.library_is_fresh(fresh)
    assert verdict and verdict.fresh is True
    assert verdict.lines == ("report is current (2 files)",)


def test_a_missing_library_is_fresh(tmp_path: Path) -> None:
    verdict = library.library_is_fresh(tmp_path / "absent")
    assert verdict and verdict.lines == ("no library",)


@pytest.mark.parametrize("with_screens", [False, True])
def test_an_empty_library_with_no_report_is_fresh(tmp_path: Path, with_screens: bool) -> None:
    tmp_path.chmod(0o755)
    if with_screens:
        (tmp_path / "screens" / "sr").mkdir(parents=True)
    verdict = library.library_is_fresh(tmp_path)
    assert verdict and verdict.lines == ("empty library, no report",)


def test_an_empty_library_with_its_report_is_fresh(tmp_path: Path) -> None:
    tmp_path.chmod(0o755)
    write_report(tmp_path)
    assert library.library_is_fresh(tmp_path).lines == ("report is current (0 files)",)


def test_a_library_with_files_and_no_report_is_refused(fresh: Path) -> None:
    (fresh / "LIBRARY-REPORT.json").unlink()
    lines = _refused(fresh)
    assert lines[0].startswith("no report: the library holds 2 files and no LIBRARY-REPORT.json")
    assert "no comparison is possible" in lines[0]
    assert lines[-1].endswith(_regeneration(fresh))
    assert "venv built from the staged payload's wheels" in lines[-1]
    assert "review the tree" in lines[-1]


@pytest.mark.parametrize("files", [None, {}, {"list": []}, {"digest": 7}, "x"])
def test_a_report_with_no_digest_is_refused(fresh: Path, files: object) -> None:
    report = json.loads((fresh / "LIBRARY-REPORT.json").read_bytes())
    if files is None:
        del report["files"]
    else:
        report["files"] = files
    _write_json(fresh, report)
    lines = _refused(fresh)
    assert lines[0].startswith("no digest: LIBRARY-REPORT.json has no files.digest")
    assert lines[-1].endswith(_regeneration(fresh))


@pytest.mark.parametrize("content", [b"", b"{", b"[]\n", b'"x"\n'])
def test_a_report_that_is_not_a_json_object_is_refused(fresh: Path, content: bytes) -> None:
    (fresh / "LIBRARY-REPORT.json").write_bytes(content)
    lines = _refused(fresh)
    assert lines[0].startswith("no report: LIBRARY-REPORT.json is not ")
    assert lines[-1].endswith(_regeneration(fresh))


def test_a_digest_mismatch_names_added_changed_and_missing_paths(fresh: Path) -> None:
    (fresh / "screens" / "a" / "new.screen.json").write_bytes(b"{}\n")
    (fresh / "screens" / "a" / "top.screen.json").write_bytes(b'{"changed": true}\n')
    (fresh / "screens" / "logo.png").unlink()
    lines = _refused(fresh)
    assert lines[0].startswith("digest mismatch: LIBRARY-REPORT.json was written for sha256:")
    assert lines[1:4] == [
        "added: screens/a/new.screen.json (in the library, not in the report)",
        "changed: screens/a/top.screen.json",
        "missing: screens/logo.png (in the report, not in the library)",
    ]
    assert lines[4].endswith(_regeneration(fresh))
    assert len(lines) == 5


def test_a_digest_mismatch_comes_with_the_listing_refusals(fresh: Path) -> None:
    """An interrupted write: a new file in place and its temporary beside it."""
    (fresh / "screens" / "a" / "new.screen.json").write_bytes(b"{}\n")
    (fresh / "screens" / "a" / ".other.screen.json.k3.tmp").write_bytes(b"{")
    lines = _refused(fresh)
    assert lines[0].startswith("leftover_tmp: screens/a/.other.screen.json.k3.tmp: ")
    assert lines[1].startswith("digest mismatch: ")
    assert lines[2] == "added: screens/a/new.screen.json (in the library, not in the report)"
    assert lines[3].endswith(_regeneration(fresh))


@pytest.mark.parametrize("listed", [None, "x", [7], [{"path": "screens/logo.png"}]])
def test_a_digest_mismatch_without_a_files_list_says_so(fresh: Path, listed: object) -> None:
    report = json.loads((fresh / "LIBRARY-REPORT.json").read_bytes())
    if listed is None:
        del report["files"]["list"]
    else:
        report["files"]["list"] = listed
    _write_json(fresh, report)
    (fresh / "screens" / "logo.png").write_bytes(b"other")
    lines = _refused(fresh)
    assert lines[0].startswith("digest mismatch: ")
    assert lines[1] == (
        "the report has no files.list, so no comparison with the library is possible"
    )
    assert lines[2].endswith(_regeneration(fresh))


def test_a_digest_that_disagrees_with_its_own_list_is_refused(fresh: Path) -> None:
    report = json.loads((fresh / "LIBRARY-REPORT.json").read_bytes())
    report["files"]["digest"] = "sha256:" + "0" * 64
    _write_json(fresh, report)
    lines = _refused(fresh)
    assert lines[1] == "the report's files.list matches the library, but its files.digest does not"


def test_a_stale_report_of_an_emptied_library_is_refused(fresh: Path) -> None:
    (fresh / "screens" / "logo.png").unlink()
    (fresh / "screens" / "a" / "top.screen.json").unlink()
    lines = _refused(fresh)
    assert [line.split(":")[0] for line in lines[1:3]] == ["missing", "missing"]


@pytest.mark.parametrize("schema", ["canopy-report/2", "osprey-canopy-write-bindings/1", "", 3])
def test_an_unknown_schema_is_refused(fresh: Path, schema: object) -> None:
    report = json.loads((fresh / "LIBRARY-REPORT.json").read_bytes())
    report["generator"]["schema"] = schema
    _write_json(fresh, report)
    lines = _refused(fresh)
    assert lines[0].startswith("unknown schema: LIBRARY-REPORT.json has generator.schema ")
    assert "canopy-library-report/1" in lines[0]
    assert lines[-1].endswith(_regeneration(fresh))
    assert len(lines) == 2


def test_the_report_of_the_previous_format_is_refused(fresh: Path) -> None:
    """A report in the earlier format carried ``format`` and no ``generator`` at all."""
    report = json.loads((fresh / "LIBRARY-REPORT.json").read_bytes())
    del report["generator"]
    report["format"] = "osprey-canopy-write-bindings/1"
    _write_json(fresh, report)
    assert _refused(fresh)[0].startswith("unknown schema: ")


def test_a_listing_refusal_is_refused_though_the_digest_matches(fresh: Path) -> None:
    (fresh / "screens" / "notes.txt").write_bytes(b"n")
    (fresh / "README.md").write_bytes(b"r")
    assert _refused(fresh) == [
        "outside_screens: README.md: "
        "a library root holds only screens/ and the LIBRARY-REPORT files; delete it",
        "suffix: screens/notes.txt: neither a *.screen.json file nor an image; delete it",
    ]


def test_a_ds_store_leaves_a_library_fresh(fresh: Path) -> None:
    (fresh / ".DS_Store").write_bytes(b"d")
    (fresh / "screens" / "a" / ".DS_Store").write_bytes(b"d")
    (fresh / "screens" / "a" / ".DS_Store").chmod(0o600)
    assert library.library_is_fresh(fresh)


def test_an_svg_is_refused_even_when_the_report_lists_it(fresh: Path) -> None:
    (fresh / "screens" / "a" / "Icon.SVG").write_bytes(b"<svg/>")
    (fresh / "screens" / "a" / "Icon.SVG").chmod(0o644)
    write_report(fresh)
    assert _refused(fresh) == [
        "svg: screens/a/Icon.SVG: SVG is markup a browser runs; "
        "delete it or convert it to a raster image"
    ]


@pytest.mark.parametrize(
    ("relpath", "mode", "said"),
    [
        ("screens/logo.png", 0o640, "mode: screens/logo.png: not readable by others (mode 0640)"),
        ("screens/logo.png", 0o646, "mode: screens/logo.png: writable by others (mode 0646)"),
        ("screens/a", 0o754, "mode: screens/a: not searchable by others (mode 0754)"),
        (
            "screens/a",
            0o750,
            "mode: screens/a: not readable by others and not searchable by others (mode 0750)",
        ),
        ("screens", 0o757, "mode: screens: writable by others (mode 0757)"),
        (".", 0o700, "mode: .: not readable by others and not searchable by others (mode 0700)"),
        (
            "LIBRARY-REPORT.json",
            0o600,
            "mode: LIBRARY-REPORT.json: not readable by others (mode 0600)",
        ),
    ],
)
def test_a_mode_the_runtime_user_cannot_use_is_refused(
    fresh: Path, relpath: str, mode: int, said: str
) -> None:
    (fresh / relpath).chmod(mode)
    try:
        assert _refused(fresh) == [said, f"fix the modes with: chmod -R o+rX,o-w {fresh}"]
    finally:
        (fresh / relpath).chmod(0o755)


def test_a_group_writable_checkout_is_not_refused(fresh: Path) -> None:
    """Only the "other" bits reach the runtime user; umask 002 leaves group-write set."""
    for path in [fresh, *fresh.rglob("*")]:
        path.chmod(0o775 if path.is_dir() else 0o664)
    assert library.library_is_fresh(fresh)


def test_the_chmod_command_repairs_what_the_mode_refusal_names(fresh: Path) -> None:
    (fresh / "screens" / "a").chmod(0o770)
    (fresh / "screens" / "logo.png").chmod(0o606)
    assert not library.library_is_fresh(fresh)
    for path in [fresh, *fresh.rglob("*")]:  # what `chmod -R o+rX,o-w` does to this tree
        mode = stat.S_IMODE(path.stat().st_mode) | stat.S_IROTH
        path.chmod((mode | (stat.S_IXOTH if path.is_dir() else 0)) & ~stat.S_IWOTH)
    assert library.library_is_fresh(fresh)


def test_the_printed_commands_quote_a_path_with_a_space(tmp_path: Path) -> None:
    root = tmp_path / "my deployment" / "library"
    assert library.regeneration_command(root) == (
        f"python -m canopy_mcp.report '{root}' --served-at library --write"
    )
    assert library.chmod_command(root) == f"chmod -R o+rX,o-w '{root}'"


def test_every_refusal_is_collected(fresh: Path) -> None:
    (fresh / "screens" / "icon.svg").write_bytes(b"<svg/>")
    (fresh / "screens" / "icon.svg").chmod(0o600)
    (fresh / "screens" / "notes.txt").write_bytes(b"n")
    kinds = [line.split(":")[0] for line in _refused(fresh)]
    assert kinds == [
        "suffix",
        "svg",
        "mode",
        "fix the modes with",
        "digest mismatch",
        "added",
        "review the tree, then regenerate the report from a venv built from the staged payload's "
        "wheels",
    ]


def test_a_path_from_the_report_cannot_start_a_line(fresh: Path) -> None:
    report = json.loads((fresh / "LIBRARY-REPORT.json").read_bytes())
    report["files"]["list"].append({"path": "screens/x\nadded: y.png", "sha256": "0" * 64})
    report["files"]["digest"] = "sha256:\n" + "0" * 64
    _write_json(fresh, report)
    lines = _refused(fresh)
    assert "sha256:\\n" in lines[0]
    assert lines[1] == ("missing: screens/x\\nadded: y.png (in the report, not in the library)")


def test_a_library_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"x")
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to("real", target_is_directory=True)
    assert _refused(tmp_path / "file")[0].startswith("not_a_library: ")
    assert "a symlink" in _refused(tmp_path / "link")[0]

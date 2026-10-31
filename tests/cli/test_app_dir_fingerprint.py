"""An application directory is build input: its files move the profile fingerprint.

``apps/<name>/`` holds the application's ``app.yml``, its fragments and its
config, all of which the build reads, so each one is folded into
``compute_profile_hash`` and named in the drift digests. The vendor files the
build fetches into the directory are its output, not its input, and are
skipped, so a fetch never moves the hash.
"""

from __future__ import annotations

from pathlib import Path

from osprey.cli.build_profile import compute_profile_hash
from osprey.deployment.staleness import profile_fingerprint

_APP_YML = """\
name: knob
vendor:
  - package: uplot
    version: 1.6.30
    file: dist/uPlot.iife.min.js
    into: panel/lib/uplot-1.6.30.min.js
    sha256: {sha}
""".format(sha="0" * 64)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root / "apps" / "knob" / "app.yml", _APP_YML)
    _write(root / "apps" / "knob" / "fragment.md", "the knob panel\n")
    return _write(root / "profile.yml", "name: Facility\n")


def test_editing_app_yml_moves_the_hash(tmp_path):
    profile = _repo(tmp_path)
    before = compute_profile_hash(profile)
    _write(profile.parent / "apps" / "knob" / "app.yml", _APP_YML + "# edited\n")
    assert compute_profile_hash(profile) != before


def test_editing_a_fragment_moves_the_hash(tmp_path):
    profile = _repo(tmp_path)
    before = compute_profile_hash(profile)
    _write(profile.parent / "apps" / "knob" / "fragment.md", "the knob panel, edited\n")
    assert compute_profile_hash(profile) != before


def test_the_drift_digests_name_the_application_directory(tmp_path):
    profile = _repo(tmp_path)
    before = profile_fingerprint(profile)
    _write(profile.parent / "apps" / "knob" / "fragment.md", "edited\n")
    after = profile_fingerprint(profile)
    assert before is not None and after is not None
    key = "material:apps/knob/"
    assert key in before.key_digests
    changed = {k for k in after.key_digests if after.key_digests[k] != before.key_digests.get(k)}
    assert key in changed


def test_vendor_files_the_build_fetches_do_not_move_the_hash(tmp_path):
    profile = _repo(tmp_path)
    before = compute_profile_hash(profile)
    digests = profile_fingerprint(profile)
    app = profile.parent / "apps" / "knob"
    _write(app / "panel" / "lib" / "uplot-1.6.30.min.js", "/* fetched */\n")
    _write(app / "panel" / "vendor" / "plotly-3.3.1.min.js", "/* fetched */\n")
    assert compute_profile_hash(profile) == before
    after = profile_fingerprint(profile)
    assert digests is not None and after is not None
    assert after.key_digests["material:apps/knob/"] == digests.key_digests["material:apps/knob/"]


def test_a_repo_without_applications_hashes_as_before(tmp_path):
    root = tmp_path / "repo"
    profile = _write(root / "profile.yml", "name: Facility\n")
    before = compute_profile_hash(profile)
    (root / "apps").mkdir()
    assert compute_profile_hash(profile) == before

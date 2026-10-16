"""The retired telemetry store is named only where a reason is written down.

OSPREY no longer deploys OpenObserve, and nothing in the tree recognises,
retires or refuses what it left behind: an old profile key is an unknown key
like any other. Its words survive only in a file whose job is to record that
the name is gone or in a frozen render today's is compared against
(:data:`ALLOWED`). Everything else that names the store is a leftover.

Two subset assertions rather than one equality:

* ``found ⊆ ALLOWED`` — nothing names the store without a reason.
* ``ALLOWED ⊆ found`` — no reason outlives the words it excuses.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The retired store's words: its name, its variables, the two variables the
#: agent read its address from, and the config and code names of its
#: organisation setting. Matched case-insensitively.
PATTERN = (
    r"openobserve|ZO_[A-Z_]+|OSPREY_TELEMETRY_(URL|ORG)|telemetry_org|store_org|telemetryUrlForRun"
)

#: Where the words are looked for. Tracked files only: a compiled cache or a
#: local render is not the tree.
SEARCHED = ("src", "packages", "plugins", "scripts", "tests", "docs", ".github")

#: Every file that names the retired store on purpose, with the reason.
ALLOWED: dict[str, str] = {
    "src/osprey/profiles/config_key_manifest.yml": (
        "the retired keys are recorded under deleted: with their orphan_sites, so "
        "none of them can come back"
    ),
    "tests/test_retired_store_words.py": "holds the pattern and both lists",
    "tests/cli/test_telemetry_gateway_emit.py": (
        "proves the old backend value is refused by the ordinary backend check"
    ),
    "tests/cli/test_telemetry_profile_keys.py": (
        "proves an old claude_code.telemetry.openobserve key is refused by the "
        "ordinary unknown-key check"
    ),
    "tests/cli/test_service_existence.py": (
        "proves a service named openobserve is refused by the ordinary unknown-service check"
    ),
    "tests/deployment/test_telemetry_config.py": (
        "asserts no port slot carries the retired service's name"
    ),
    "tests/dispatch/test_dashboard_config_injection.py": (
        "asserts the dashboard no longer builds a link to the retired store"
    ),
    "tests/deployment/web_terminals/golden/pre_audit_roles/docker-compose.web.yml": (
        "a frozen render that predates the audit-and-roles feature, compared line "
        "by line and never regenerated"
    ),
    "tests/deployment/web_terminals/test_auth_off_baseline_pin.py": (
        "names the two carrier lines the frozen baseline has and today's render must not"
    ),
}


def _tracked_files() -> list[str]:
    """Every file git knows under :data:`SEARCHED`, tracked or new and not ignored."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SEARCHED],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return [name for name in listed.stdout.decode("utf-8").split("\0") if name]


def files_naming_the_retired_store() -> set[str]:
    """The searched files whose text matches :data:`PATTERN`."""
    pattern = re.compile(PATTERN, re.IGNORECASE)
    found: set[str] = set()
    for name in _tracked_files():
        path = REPO_ROOT / name
        if not path.is_file():
            continue  # deleted in the working tree
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # a binary file names nothing
        if pattern.search(text):
            found.add(name)
    return found


def test_nothing_names_the_retired_store_without_a_reason() -> None:
    strays = sorted(files_naming_the_retired_store() - set(ALLOWED))
    assert strays == [], (
        "file(s) name the retired telemetry store with no entry in ALLOWED — "
        "remove the words, or add the file with its reason:\n  " + "\n  ".join(strays)
    )


def test_every_reason_still_has_its_words() -> None:
    stale = sorted(set(ALLOWED) - files_naming_the_retired_store())
    assert stale == [], (
        "ALLOWED entries whose file no longer names the retired store — drop the entry:\n  "
        + "\n  ".join(stale)
    )


def test_every_reason_is_written_down() -> None:
    assert all(reason.strip() for reason in ALLOWED.values())


def test_the_pattern_matches_each_retired_word() -> None:
    pattern = re.compile(PATTERN, re.IGNORECASE)
    for word in (
        "OpenObserve",
        "ZO_ROOT_USER_PASSWORD",
        "ZO_INGEST_SA_TOKEN",
        "OSPREY_TELEMETRY_URL",
        "OSPREY_TELEMETRY_ORG",
        "telemetry_org",
        "store_org",
        "telemetryUrlForRun",
    ):
        assert pattern.search(word), word
    for word in ("telemetry_store", "ZOOM", "OSPREY_TELEMETRY_DIGEST_STORE"):
        assert not pattern.search(word), word

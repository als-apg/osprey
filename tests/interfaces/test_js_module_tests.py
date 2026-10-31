"""Every production JavaScript module has its own Vitest file, or is listed.

The rule: a module ``foo.js`` under ``src/osprey/interfaces/<interface>/`` is
covered by a tracked ``foo.test.js`` or ``foo.test.mjs`` that sits either
beside the module, or at the mirrored directory below
``tests/interfaces/<interface>/``. The module's directory is taken below its
JS root, ``static/js/`` or ``panels/<name>/``; the test's directory is taken
below ``tests/interfaces/<interface>/`` with one optional leading ``js/``
dropped. So ``static/js/scaffold/utils.js`` is covered by
``scaffold/utils.test.mjs`` or ``js/scaffold/utils.test.mjs`` there, and
``static/js/utils.js`` by neither. A module under no JS root is covered only by
a test beside it, and a guard test names every such module.

Matching is per interface, not by basename across the repository, because
basenames repeat: ``app``, ``helpers``, ``render`` and ``settings`` each name a
module in more than one interface, and a test for one of them says nothing
about the others.

``WITHOUT_OWN_TEST`` lists the modules that have no such file. The table only
shrinks: these tests fail when a new module arrives without its test, and they
fail when a listed module gains a test or disappears while its entry stays.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]

INTERFACES_ROOT = "src/osprey/interfaces/"
TESTS_ROOT = "tests/interfaces/"

#: Suffixes a Vitest file may carry after the module's stem.
TEST_SUFFIXES = (".test.js", ".test.mjs")

#: The JS roots, matched at the start of a module's path below its interface.
JS_ROOT = re.compile(r"(?:static/js|panels/[^/]+)/")

#: The optional first directory below ``tests/interfaces/<interface>/``.
TESTS_JS_DIR = "js"

#: Modules with no same-named Vitest file in their own interface. A path leaves
#: this table in the change that adds its test or deletes the module; nothing
#: is added except by an owner-approved change.
WITHOUT_OWN_TEST: tuple[str, ...] = (
    "src/osprey/interfaces/ariel/static/js/api.js",
    "src/osprey/interfaces/ariel/static/js/app.js",
    "src/osprey/interfaces/ariel/static/js/components.js",
    "src/osprey/interfaces/ariel/static/js/dashboard.js",
    "src/osprey/interfaces/ariel/static/js/entries-detail.js",
    "src/osprey/interfaces/ariel/static/js/entries-form.js",
    "src/osprey/interfaces/ariel/static/js/entries.js",
    "src/osprey/interfaces/ariel/static/js/search.js",
    "src/osprey/interfaces/ariel/static/js/settings.js",
    "src/osprey/interfaces/ariel/static/js/utils.js",
    "src/osprey/interfaces/artifacts/static/js/logbook-template.js",
    "src/osprey/interfaces/artifacts/static/js/md-render.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/figure-controls.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/hyperscript.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/plan-browser.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/plan-presentation.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/plans-view.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/queue-client.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/queue-view.js",
    "src/osprey/interfaces/bluesky_web/panels/bluesky/results-view.js",
    "src/osprey/interfaces/channel_finder/static/js/api.js",
    "src/osprey/interfaces/channel_finder/static/js/app.js",
    "src/osprey/interfaces/channel_finder/static/js/explore-hierarchical.js",
    "src/osprey/interfaces/channel_finder/static/js/explore-in-context.js",
    "src/osprey/interfaces/channel_finder/static/js/explore-middle-layer.js",
    "src/osprey/interfaces/channel_finder/static/js/explore.js",
    "src/osprey/interfaces/channel_finder/static/js/feedback-detail.js",
    "src/osprey/interfaces/channel_finder/static/js/feedback-state.js",
    "src/osprey/interfaces/channel_finder/static/js/feedback.js",
    "src/osprey/interfaces/channel_finder/static/js/graph-finder-data.js",
    "src/osprey/interfaces/channel_finder/static/js/graph-finder-events.js",
    "src/osprey/interfaces/channel_finder/static/js/graph-finder-verbs.js",
    "src/osprey/interfaces/channel_finder/static/js/modal.js",
    "src/osprey/interfaces/channel_finder/static/js/state.js",
    "src/osprey/interfaces/channel_finder/static/js/utils.js",
    "src/osprey/interfaces/design_system/static/js/check-name.js",
    "src/osprey/interfaces/design_system/static/js/components/osprey-display-menu.js",
    "src/osprey/interfaces/design_system/static/js/components/osprey-drawer.js",
    "src/osprey/interfaces/design_system/static/js/components/osprey-theme-switcher.js",
    "src/osprey/interfaces/design_system/static/js/platform.js",
    "src/osprey/interfaces/design_system/static/js/theme-lab-ui.js",
    "src/osprey/interfaces/design_system/static/js/theme-lab.js",
    "src/osprey/interfaces/design_system/static/js/theme-manager.js",
    "src/osprey/interfaces/design_system/static/js/tokens.js",
    "src/osprey/interfaces/design_system/static/js/uuid.js",
    "src/osprey/interfaces/health/static/js/dashboard.js",
    "src/osprey/interfaces/lattice_dashboard/static/js/app.js",
    "src/osprey/interfaces/okf_panel/static/js/app.js",
    "src/osprey/interfaces/web_terminal/static/js/activity-format.js",
    "src/osprey/interfaces/web_terminal/static/js/activity-history.js",
    "src/osprey/interfaces/web_terminal/static/js/app.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-activity.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-blocks.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-dom.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-features.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-icons.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-permission.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-question.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-system.js",
    "src/osprey/interfaces/web_terminal/static/js/chat-tool-vocab.js",
    "src/osprey/interfaces/web_terminal/static/js/config-renderers.js",
    "src/osprey/interfaces/web_terminal/static/js/confirm-skip.js",
    "src/osprey/interfaces/web_terminal/static/js/diff-utils.js",
    "src/osprey/interfaces/web_terminal/static/js/dock-tab.js",
    "src/osprey/interfaces/web_terminal/static/js/dock-workspace.js",
    "src/osprey/interfaces/web_terminal/static/js/logout.js",
    "src/osprey/interfaces/web_terminal/static/js/modal-overlay.js",
    "src/osprey/interfaces/web_terminal/static/js/palette-boot.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-agent-attention.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-commands.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-empty-state.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-health.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-lifecycle.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-placement.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-queries.js",
    "src/osprey/interfaces/web_terminal/static/js/panel-sse.js",
    "src/osprey/interfaces/web_terminal/static/js/scaffold-gallery.js",
    "src/osprey/interfaces/web_terminal/static/js/scaffold/cards.js",
    "src/osprey/interfaces/web_terminal/static/js/scaffold/detail-content.js",
    "src/osprey/interfaces/web_terminal/static/js/scaffold/edit-form.js",
    "src/osprey/interfaces/web_terminal/static/js/scaffold/write-gate.js",
    "src/osprey/interfaces/web_terminal/static/js/session.js",
    "src/osprey/interfaces/web_terminal/static/js/settings-apply-confirm.js",
    "src/osprey/interfaces/web_terminal/static/js/settings-notice.js",
    "src/osprey/interfaces/web_terminal/static/js/settings-protected.js",
    "src/osprey/interfaces/web_terminal/static/js/svg-icons.js",
    "src/osprey/interfaces/web_terminal/static/js/terminal.js",
    "src/osprey/interfaces/web_terminal/static/js/undo-notice.js",
)


def tracked_files() -> list[str]:
    """Every file of the working tree git tracks or would track, repo-relative.

    Tracked files deleted from the tree are left out and untracked files that
    are not ignored are kept in, so a module or test that is renamed, added or
    removed is judged before the change is committed.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    return sorted({path for path in out.split("\0") if path and (REPO_ROOT / path).is_file()})


def modules(files: Iterable[str] | None = None) -> list[str]:
    """The production JavaScript modules among ``files`` (default: tracked files)."""
    if files is None:
        files = tracked_files()
    return sorted(
        path
        for path in files
        if path.startswith(INTERFACES_ROOT)
        and path.endswith(".js")
        and "/vendor/" not in path
        and not path.endswith(".min.js")
        and ".test." not in path
    )


def interface_of(module: str) -> str:
    """The interface a module belongs to: the path segment after the interfaces root."""
    return module[len(INTERFACES_ROOT) :].split("/", 1)[0]


def module_directory(module: str) -> str | None:
    """A module's directory below its JS root ("" at the root), ``None`` under no root."""
    below = module[len(INTERFACES_ROOT) :].split("/", 1)[1]
    root = JS_ROOT.match(below)
    if root is None:
        return None
    return "/".join(PurePosixPath(below[root.end() :]).parent.parts)


def _test_directory(test: str, interface: str) -> str | None:
    """A test's directory below its interface's test root, one leading ``js`` dropped."""
    tests_root = f"{TESTS_ROOT}{interface}/"
    if not test.startswith(tests_root):
        return None
    parts = PurePosixPath(test[len(tests_root) :]).parent.parts
    if parts[:1] == (TESTS_JS_DIR,):
        parts = parts[1:]
    return "/".join(parts)


def covering_tests(module: str, files: Iterable[str] | None = None) -> list[str]:
    """The Vitest files among ``files`` named for ``module``, beside it or mirrored."""
    if files is None:
        files = tracked_files()
    interface = interface_of(module)
    module_path = PurePosixPath(module)
    names = {module_path.stem + suffix for suffix in TEST_SUFFIXES}
    directory = module_directory(module)

    def covers(path: str) -> bool:
        test = PurePosixPath(path)
        if test.name not in names:
            return False
        if test.parent == module_path.parent:
            return True
        return directory is not None and _test_directory(path, interface) == directory

    return sorted(path for path in files if covers(path))


def expected_test_path(module: str) -> str:
    """Where a module's own Vitest file is expected by default."""
    directory = module_directory(module) or ""
    below = f"{directory}/" if directory else ""
    return f"{TESTS_ROOT}{interface_of(module)}/{below}{PurePosixPath(module).stem}.test.mjs"


def uncovered_unlisted(files: Iterable[str], listed: Iterable[str]) -> list[str]:
    """Modules with neither their own test nor a table entry."""
    files = list(files)
    listed = set(listed)
    return [m for m in modules(files) if not covering_tests(m, files) and m not in listed]


def listed_missing(files: Iterable[str], listed: Iterable[str]) -> list[str]:
    """Table entries that name no module."""
    present = set(modules(files))
    return [m for m in listed if m not in present]


def listed_with_test(files: Iterable[str], listed: Iterable[str]) -> list[str]:
    """Table entries whose module has its own test."""
    files = list(files)
    return [m for m in listed if covering_tests(m, files)]


def test_every_module_has_its_own_test_or_is_listed():
    missing = uncovered_unlisted(tracked_files(), WITHOUT_OWN_TEST)
    assert missing == [], "modules without their own Vitest file:\n" + "\n".join(
        f"  {m} -> add {expected_test_path(m)}" for m in missing
    )


def test_every_listed_module_exists():
    gone = listed_missing(tracked_files(), WITHOUT_OWN_TEST)
    assert gone == [], "remove from WITHOUT_OWN_TEST, the module is gone:\n" + "\n".join(
        f"  {m}" for m in gone
    )


def test_no_listed_module_has_gained_a_test():
    files = tracked_files()
    covered = listed_with_test(files, WITHOUT_OWN_TEST)
    assert covered == [], "remove from WITHOUT_OWN_TEST, the module has a test:\n" + "\n".join(
        f"  {m} <- {', '.join(covering_tests(m, files))}" for m in covered
    )


def test_the_table_is_sorted_and_unique():
    assert list(WITHOUT_OWN_TEST) == sorted(set(WITHOUT_OWN_TEST))


WT_SETTINGS = "src/osprey/interfaces/web_terminal/static/js/settings.js"
WT_SETTINGS_TEST = "tests/interfaces/web_terminal/settings.test.mjs"
ARIEL_SETTINGS_TEST = "tests/interfaces/ariel/settings.test.mjs"


def test_vendored_minified_and_test_files_are_not_modules():
    files = [
        WT_SETTINGS,
        "src/osprey/interfaces/web_terminal/static/js/vendor/xterm.js",
        "src/osprey/interfaces/okf_panel/static/vendor/marked.min.js",
        "src/osprey/interfaces/health/static/js/helpers.test.js",
        "src/osprey/interfaces/web_terminal/static/css/app.css",
        "tests/interfaces/web_terminal/js/tour.test.js",
        "scripts/build.js",
    ]
    assert modules(files) == [WT_SETTINGS]


def test_a_new_uncovered_module_is_reported():
    new = "src/osprey/interfaces/web_terminal/static/js/brand-new.js"
    files = [WT_SETTINGS, WT_SETTINGS_TEST, new]
    assert uncovered_unlisted(files, listed=()) == [new]
    assert expected_test_path(new) == "tests/interfaces/web_terminal/brand-new.test.mjs"


def test_a_listed_uncovered_module_is_not_reported():
    assert uncovered_unlisted([WT_SETTINGS], listed=(WT_SETTINGS,)) == []


def test_a_listed_module_that_gained_a_test_is_reported():
    files = [WT_SETTINGS, WT_SETTINGS_TEST]
    assert listed_with_test(files, listed=(WT_SETTINGS,)) == [WT_SETTINGS]
    assert listed_with_test([WT_SETTINGS], listed=(WT_SETTINGS,)) == []


def test_a_listed_module_that_was_deleted_is_reported():
    assert listed_missing([WT_SETTINGS_TEST], listed=(WT_SETTINGS,)) == [WT_SETTINGS]


def test_a_same_named_test_in_another_interface_does_not_cover():
    files = [WT_SETTINGS, ARIEL_SETTINGS_TEST]
    assert covering_tests(WT_SETTINGS, files) == []
    assert uncovered_unlisted(files, listed=()) == [WT_SETTINGS]


def test_a_test_beside_the_module_or_under_js_covers():
    beside = "src/osprey/interfaces/health/static/js/helpers.test.mjs"
    nested = "tests/interfaces/web_terminal/js/settings.test.js"
    helpers = "src/osprey/interfaces/health/static/js/helpers.js"
    assert covering_tests(helpers, [helpers, beside]) == [beside]
    assert covering_tests(WT_SETTINGS, [WT_SETTINGS, nested]) == [nested]


def test_a_test_named_for_a_longer_stem_does_not_cover():
    files = [WT_SETTINGS, "tests/interfaces/web_terminal/settings-banner.test.mjs"]
    assert covering_tests(WT_SETTINGS, files) == []


WT_UTILS = "src/osprey/interfaces/web_terminal/static/js/utils.js"
WT_SCAFFOLD_UTILS = "src/osprey/interfaces/web_terminal/static/js/scaffold/utils.js"
WT_UTILS_TEST = "tests/interfaces/web_terminal/utils.test.mjs"
WT_SCAFFOLD_UTILS_TEST = "tests/interfaces/web_terminal/scaffold/utils.test.mjs"


def test_a_same_named_test_in_a_sibling_directory_does_not_cover():
    files = [WT_UTILS, WT_SCAFFOLD_UTILS_TEST]
    assert covering_tests(WT_UTILS, files) == []
    assert uncovered_unlisted(files, listed=()) == [WT_UTILS]
    assert covering_tests(WT_SCAFFOLD_UTILS, [WT_SCAFFOLD_UTILS, WT_SCAFFOLD_UTILS_TEST]) == [
        WT_SCAFFOLD_UTILS_TEST
    ]
    assert covering_tests(WT_SCAFFOLD_UTILS, [WT_SCAFFOLD_UTILS, WT_UTILS_TEST]) == []


def test_a_panel_module_is_covered_at_the_interface_test_root():
    module = "src/osprey/interfaces/bluesky_web/panels/bluesky/x.js"
    test = "tests/interfaces/bluesky_web/x.test.mjs"
    assert covering_tests(module, [module, test]) == [test]


def test_a_nested_module_expects_its_test_in_the_mirrored_directory():
    module = "src/osprey/interfaces/web_terminal/static/js/scaffold/new.js"
    assert expected_test_path(module) == "tests/interfaces/web_terminal/scaffold/new.test.mjs"


def test_every_module_sits_under_a_js_root():
    rootless = [m for m in modules() if module_directory(m) is None]
    assert rootless == [], (
        "modules outside every JS root; add their location to JS_ROOT:\n"
        + "\n".join(f"  {m}" for m in rootless)
    )


def test_a_module_outside_every_js_root_is_rootless():
    module = "src/osprey/interfaces/x/static/other.js"
    assert module_directory(module) is None
    assert module_directory("src/osprey/interfaces/x/static/js/a/b.js") == "a"
    assert module_directory("src/osprey/interfaces/x/panels/p/b.js") == ""
    beside = "src/osprey/interfaces/x/static/other.test.mjs"
    elsewhere = "tests/interfaces/x/other.test.mjs"
    assert covering_tests(module, [module, beside, elsewhere]) == [beside]

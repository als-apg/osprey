"""Tests for the static path policy that refuses render-zone writes.

The policy is mode-independent: the same literal write is an issue whether the
run is readonly or readwrite, because the boundary it defends is the agent
changing its own configuration, not a control-system write. What the walker
must *not* do is guess — dynamic paths belong to the runtime guard, and this
suite pins that silence as firmly as it pins the refusals.
"""

import pytest

from osprey.services.python_executor.analysis.safety_checks import quick_safety_check
from osprey.services.python_executor.execution.path_policy import path_policy_issues


@pytest.fixture
def roots(tmp_path):
    """Render zone + profile source protected; the agent's data zone carved out."""
    return {
        "protected_roots": [tmp_path / "build", tmp_path / "profile.yml"],
        "permitted_roots": [tmp_path / "var" / "agent_data"],
        "working_directory": tmp_path,
    }


def test_render_zone_write_is_flagged(roots):
    issues = path_policy_issues("open('build/config.yml', 'w')", **roots)
    assert len(issues) == 1
    assert "build/config.yml" in issues[0]


def test_profile_source_write_is_flagged(roots):
    issues = path_policy_issues("open('profile.yml', 'w')", **roots)
    assert len(issues) == 1
    assert "profile.yml" in issues[0]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("open('var/agent_data/x', 'w')", id="agent-data-write"),
        pytest.param("open('build/config.yml', mode)", id="dynamic-mode"),
        pytest.param("open('/tmp/scratch.txt', 'w')", id="absolute-path-outside-every-root"),
        pytest.param("Path('var/agent_data/x').write_text('y')", id="path-write-in-agent-data"),
        pytest.param("Path('var/agent_data/x').open('w')", id="path-open-in-agent-data"),
        pytest.param(
            "shutil.copy('build/config.yml', 'var/agent_data/copy.yml')",
            id="copy-from-render-zone-into-agent-data",
        ),
        pytest.param("os.makedirs('var/agent_data/figures')", id="makedirs-in-agent-data"),
        pytest.param("shutil.rmtree(target_dir)", id="rmtree-of-dynamic-dir"),
        pytest.param("os.remove(f'build/{name}')", id="remove-of-dynamic-path"),
        pytest.param("open('build/x', 'w'", id="syntax-error"),
        pytest.param("open('../build/config.yml', 'w')", id="parent-escaping-relative-path"),
        # Near-misses of the sandbox-guard identifiers; the matching rule is
        # described under "Sandbox-guard tampering" below.
        pytest.param("my_osprey_fsx = 1", id="embedded-not-prefixed"),
        pytest.param("osprey_fs = load()", id="no-leading-underscore"),
        pytest.param("x = df._osprey", id="shorter-than-prefix"),
        pytest.param("restore_patched_targets()", id="near-miss-exact-name"),
        pytest.param("_restore_patched_targets_extra = 1", id="exact-name-is-not-a-prefix"),
        pytest.param("'the _osprey_fs guard rejects this path'", id="prose-mentioning-a-name"),
        # Reloads that cannot disarm the guard; see "Reloading a guarded module" below.
        pytest.param("import importlib\nimportlib.reload(numpy)", id="reload-unguarded-module"),
        pytest.param("config.reload()", id="reload-method-named-reload"),
        pytest.param("reload()", id="reload-no-argument"),
        pytest.param("os.remove('scratch.txt')", id="ordinary-os-call"),
    ],
)
def test_a_write_the_policy_must_not_guess_at_stays_clean(code, roots):
    assert path_policy_issues(code, **roots) == []


def test_read_mode_open_is_clean(roots):
    """A read is never an issue — the default mode included."""
    assert path_policy_issues("open('file.txt')", **roots) == []
    assert path_policy_issues("open('build/config.yml', 'r')", **roots) == []
    assert path_policy_issues("open('build/config.yml', 'rb')", **roots) == []


def test_quick_safety_check_still_passes_a_plain_read():
    """Regression: the existing checker was left untouched by this policy."""
    passed, issues = quick_safety_check("open('file.txt')")
    assert passed, issues


def test_dynamic_path_is_deferred_to_runtime(roots):
    code = "target = 'build/config.yml'\nopen(target, 'w')\n"
    assert path_policy_issues(code, **roots) == []


def test_keyword_spelled_open_is_flagged(roots):
    issues = path_policy_issues("open(file='build/config.yml', mode='w')", **roots)
    assert len(issues) == 1


def test_absolute_path_into_render_zone_is_flagged(tmp_path, roots):
    issues = path_policy_issues(f"open({str(tmp_path / 'build' / 'x.yml')!r}, 'w')", **roots)
    assert len(issues) == 1


def test_io_open_write_is_flagged(roots):
    assert len(path_policy_issues("io.open('build/config.yml', 'w')", **roots)) == 1
    assert path_policy_issues("io.open('build/config.yml')", **roots) == []


@pytest.mark.parametrize(
    "code",
    [
        "Path('build/config.yml').write_text('x')",
        "Path('build/config.yml').write_bytes(b'x')",
        "Path('build') / 'config.yml'; Path('build', 'config.yml').write_text('x')",
        "pathlib.Path('build/config.yml').open('w')",
        "shutil.copy('src.yml', 'build/config.yml')",
        "shutil.copy2('src.yml', 'build/config.yml')",
        "shutil.copyfile('src.yml', 'build/config.yml')",
        "shutil.move('build/config.yml', 'var/agent_data/config.yml')",
        "shutil.rmtree('build')",
        "os.remove('build/config.yml')",
        "os.unlink('build/config.yml')",
        "os.rename('build/config.yml', 'var/agent_data/x')",
        "os.replace('var/agent_data/x', 'build/config.yml')",
        "os.makedirs('build/data')",
        "os.mkdir('build/data')",
        "os.truncate('build/config.yml', 0)",
    ],
)
def test_every_covered_write_spelling_is_flagged(code, roots):
    issues = path_policy_issues(code, **roots)
    assert issues, f"no issue raised for: {code}"


def test_path_read_methods_are_clean(roots):
    assert path_policy_issues("Path('build/config.yml').read_text()", **roots) == []
    assert path_policy_issues("Path('build/config.yml').open('r')", **roots) == []


def test_aliased_module_is_still_seen(roots):
    """Name matching, not module binding — ``import shutil as sh`` still counts."""
    assert path_policy_issues("sh.rmtree('build')", **roots)


def test_no_protected_roots_means_no_issues(tmp_path):
    assert (
        path_policy_issues(
            "open('build/config.yml', 'w')", protected_roots=[], working_directory=tmp_path
        )
        == []
    )


def test_every_write_in_a_script_is_reported(roots):
    code = "open('build/a.yml', 'w')\nopen('build/b.yml', 'w')\nopen('var/agent_data/c', 'w')\n"
    assert len(path_policy_issues(code, **roots)) == 2


# ---------------------------------------------------------------------------
# Sandbox-guard tampering
#
# Matching rule, pinned in both directions below: an identifier is the guard's
# when it *starts with* ``_osprey_fs`` or ``_OSPREY_FS`` (case-sensitive), or
# is exactly ``_restore_patched_targets``. String constants count too, but only
# when the whole string is an identifier — otherwise prose that mentions a
# guard name would refuse a legitimate script.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "message"),
    [
        pytest.param("_restore_patched_targets()", "sandbox guard", id="exact-name"),
        pytest.param("_osprey_fs_check('x', True)", "sandbox guard", id="helper-prefix"),
        pytest.param("x = _OSPREY_FS_PROTECTED", "sandbox guard", id="table-prefix"),
        pytest.param(
            "import builtins\nbuiltins._osprey_fs_originals", "sandbox guard", id="attribute"
        ),
        pytest.param("getattr(m, '_osprey_fs_check')", "sandbox guard", id="string-literal"),
        pytest.param(
            "globals()['_restore_patched_targets']", "sandbox guard", id="globals-subscript"
        ),
        pytest.param(
            "def _osprey_fs_under(p, r):\n    return False", "sandbox guard", id="shadow-def"
        ),
        pytest.param("import mod as _OSPREY_FS_TARGETS", "sandbox guard", id="import-alias"),
        # Reloading a guarded module: see "Reloading a guarded module" below.
        pytest.param(
            "import importlib\nimportlib.reload(os)", "reload", id="reload-importlib-dotted"
        ),
        pytest.param("from importlib import reload\nreload(shutil)", "reload", id="reload-bare"),
        pytest.param("importlib.reload(builtins)", "reload", id="reload-builtins"),
        pytest.param("importlib.reload(io)", "reload", id="reload-io"),
        pytest.param(
            "importlib.reload(sys.modules['os'])", "reload", id="reload-sys-modules-subscript"
        ),
        pytest.param("imp.reload(os)", "reload", id="reload-any-receiver"),
    ],
)
def test_guard_tampering_is_flagged(code, message, roots):
    issues = path_policy_issues(code, **roots)
    assert issues, f"no issue raised for: {code}"
    assert message in issues[0]


def test_guard_tamper_is_flagged_without_any_protected_root(tmp_path):
    """Guard ownership does not depend on the path sets — there is no path."""
    assert path_policy_issues(
        "_restore_patched_targets()", protected_roots=[], working_directory=tmp_path
    )


def test_each_distinct_guard_name_is_reported_once(tmp_path):
    code = "_osprey_fs_check(_OSPREY_FS_PROTECTED)\n_osprey_fs_check(1)\n"
    issues = path_policy_issues(code, protected_roots=[], working_directory=tmp_path)
    assert len(issues) == 2


# ---------------------------------------------------------------------------
# Reloading a guarded module
#
# ``importlib.reload(os)`` re-executes the module and rebinds every name in it
# from the original C primitives — the runtime guard's patches go with them.
# It is a plainer spelling of the disarm than the ``getattr`` form above, and
# the runtime guard cannot see it at all (nothing calls a patched entry point),
# so this walker is the only layer that can refuse it.
# ---------------------------------------------------------------------------


def test_reload_is_flagged_without_any_protected_root(tmp_path):
    """Same as guard-identifier tampering: it does not depend on the path sets."""
    assert path_policy_issues(
        "importlib.reload(os)", protected_roots=[], working_directory=tmp_path
    )


def test_reloading_a_guarded_module_is_flagged__mutation_renames_the_module(roots):
    """Proves the check keys on the guarded module, not on the word 'reload'."""
    with pytest.raises(AssertionError):
        assert path_policy_issues("importlib.reload(os_lookalike)", **roots)


# ---------------------------------------------------------------------------
# Relative literals resolve against the child's working directory
# ---------------------------------------------------------------------------


def test_a_relative_literal_resolves_against_the_working_directory(tmp_path, roots):
    elsewhere = tmp_path / "elsewhere"
    assert (
        path_policy_issues(
            "open('build/config.yml', 'w')", **{**roots, "working_directory": elsewhere}
        )
        == []
    )
    issues = path_policy_issues(
        "open('../build/config.yml', 'w')", **{**roots, "working_directory": elsewhere}
    )
    assert len(issues) == 1
    assert "../build/config.yml" in issues[0]


def test_a_nested_entry_does_not_refuse_an_unrelated_relative_write(tmp_path):
    """No trailing-component match: ``plans/`` here is not ``nested/deep/plans``."""
    protected = [tmp_path / "nested" / "deep" / "plans", tmp_path.parent / "shared" / "plans"]
    assert (
        path_policy_issues(
            "open('plans/scan.py', 'w')", protected_roots=protected, working_directory=tmp_path
        )
        == []
    )
    assert path_policy_issues(
        "open('nested/deep/plans/scan.py', 'w')",
        protected_roots=protected,
        working_directory=tmp_path,
    )


# ---------------------------------------------------------------------------
# Removing or relocating a directory that contains a protected path
# ---------------------------------------------------------------------------


@pytest.fixture
def entry_roots(tmp_path):
    """A directory entry and a file entry, both one level below ``lib/``."""
    return {
        "protected_roots": [tmp_path / "lib" / "plans", tmp_path / "lib" / "seed.ttl"],
        "permitted_roots": [tmp_path / "var" / "agent_data"],
        "working_directory": tmp_path,
    }


@pytest.mark.parametrize(
    "code",
    [
        "shutil.rmtree('lib')",
        "shutil.rmtree('.')",
        "os.rmdir('lib')",
        "os.removedirs('lib')",
        "os.remove('lib')",
        "os.rename('lib', 'var/agent_data/lib')",
        "os.replace('var/agent_data/prep', 'lib')",
        "shutil.move('var/agent_data/prep', 'lib')",
        "os.symlink('var/agent_data/prep', 'lib')",
        "os.link('var/agent_data/prep', 'lib')",
        "Path('lib').rename('var/agent_data/lib')",
        "Path('var/agent_data/prep').replace('lib')",
        "Path('lib').rmdir()",
    ],
)
def test_removing_or_relocating_the_parent_of_an_entry_is_flagged(code, entry_roots):
    issues = path_policy_issues(code, **entry_roots)
    assert issues, f"no issue raised for: {code}"
    assert "contains the protected location" in issues[0]


@pytest.mark.parametrize(
    "code",
    [
        "os.makedirs('lib')",
        "os.mkdir('lib')",
        "Path('lib').mkdir()",
        "open('lib/notes.txt', 'w')",
        "shutil.copy('var/agent_data/x', 'lib')",
        "shutil.rmtree('var/agent_data/lib')",
    ],
)
def test_creating_the_parent_of_an_entry_and_copying_into_it_stay_clean(code, entry_roots):
    assert path_policy_issues(code, **entry_roots) == []


# ---------------------------------------------------------------------------
# Shared method names: only a path module or a path object is a path call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("names = ['profile.yml']\nnames.remove('profile.yml')", id="list.remove"),
        pytest.param("'profile.yml'.replace('profile.yml', 'build')", id="str.replace-literal"),
        pytest.param("text.replace('build/config.yml', 'profile.yml')", id="str.replace"),
        pytest.param("frame.rename('profile.yml', 'build')", id="dataframe.rename"),
        pytest.param("settings.copy('build')", id="object.copy"),
        pytest.param("cursor.move('build', 'profile.yml')", id="object.move"),
        pytest.param("remove('profile.yml')", id="bare-unimported-name"),
    ],
)
def test_list_and_string_methods_with_an_entry_name_stay_clean(code, roots):
    assert path_policy_issues(code, **roots) == []


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("os.remove('profile.yml')", id="os"),
        pytest.param("import os as o\no.remove('profile.yml')", id="module-alias"),
        pytest.param("from os import remove\nremove('profile.yml')", id="from-import"),
        pytest.param("from os import replace as swap\nswap('x', 'profile.yml')", id="from-alias"),
        pytest.param("import shutil as sh\nsh.move('build', 'x')", id="shutil-alias"),
        pytest.param("posix.rename('x', 'profile.yml')", id="posix"),
        pytest.param("Path('x').replace('profile.yml')", id="path-object"),
        pytest.param("(Path('build') / 'config.yml').rename('x')", id="joined-path-object"),
        pytest.param("shutil.copy('x', 'build/config.yml')", id="shutil.copy"),
    ],
)
def test_shared_names_on_a_path_module_or_object_are_flagged(code, roots):
    assert path_policy_issues(code, **roots)

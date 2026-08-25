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

pytestmark = pytest.mark.unit


@pytest.fixture
def roots(tmp_path):
    """Render zone + profile source protected; the agent's data zone carved out."""
    return {
        "protected_roots": [tmp_path / "build", tmp_path / "profile.yml"],
        "permitted_roots": [tmp_path / "var" / "agent_data"],
    }


def test_render_zone_write_is_flagged(roots):
    issues = path_policy_issues("open('build/config.yml', 'w')", **roots)
    assert len(issues) == 1
    assert "build/config.yml" in issues[0]


def test_profile_source_write_is_flagged(roots):
    issues = path_policy_issues("open('profile.yml', 'w')", **roots)
    assert len(issues) == 1
    assert "profile.yml" in issues[0]


def test_agent_data_write_is_clean(roots):
    assert path_policy_issues("open('var/agent_data/x', 'w')", **roots) == []


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


def test_dynamic_mode_is_deferred_to_runtime(roots):
    assert path_policy_issues("open('build/config.yml', mode)", **roots) == []


def test_keyword_spelled_open_is_flagged(roots):
    issues = path_policy_issues("open(file='build/config.yml', mode='w')", **roots)
    assert len(issues) == 1


def test_absolute_path_into_render_zone_is_flagged(tmp_path, roots):
    issues = path_policy_issues(f"open({str(tmp_path / 'build' / 'x.yml')!r}, 'w')", **roots)
    assert len(issues) == 1


def test_absolute_path_outside_every_root_is_clean(roots):
    assert path_policy_issues("open('/tmp/scratch.txt', 'w')", **roots) == []


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


@pytest.mark.parametrize(
    "code",
    [
        "Path('var/agent_data/x').write_text('y')",
        "Path('var/agent_data/x').open('w')",
        "shutil.copy('build/config.yml', 'var/agent_data/copy.yml')",
        "os.makedirs('var/agent_data/figures')",
        "shutil.rmtree(target_dir)",
        "os.remove(f'build/{name}')",
    ],
)
def test_permitted_and_dynamic_spellings_stay_clean(code, roots):
    assert path_policy_issues(code, **roots) == []


def test_path_read_methods_are_clean(roots):
    assert path_policy_issues("Path('build/config.yml').read_text()", **roots) == []
    assert path_policy_issues("Path('build/config.yml').open('r')", **roots) == []


def test_aliased_module_is_still_seen(roots):
    """Name matching, not module binding — ``import shutil as sh`` still counts."""
    assert path_policy_issues("sh.rmtree('build')", **roots)


def test_syntax_error_stays_quiet(roots):
    assert path_policy_issues("open('build/x', 'w'", **roots) == []


def test_no_protected_roots_means_no_issues(tmp_path):
    assert path_policy_issues("open('build/config.yml', 'w')", protected_roots=[]) == []


def test_parent_escaping_relative_path_is_deferred(roots):
    assert path_policy_issues("open('../build/config.yml', 'w')", **roots) == []


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
    "code",
    [
        pytest.param("_restore_patched_targets()", id="exact-name"),
        pytest.param("_osprey_fs_check('x', True)", id="helper-prefix"),
        pytest.param("x = _OSPREY_FS_PROTECTED", id="table-prefix"),
        pytest.param("import builtins\nbuiltins._osprey_fs_originals", id="attribute"),
        pytest.param("getattr(m, '_osprey_fs_check')", id="string-literal"),
        pytest.param("globals()['_restore_patched_targets']", id="globals-subscript"),
        pytest.param("def _osprey_fs_under(p, r):\n    return False", id="shadow-def"),
        pytest.param("import mod as _OSPREY_FS_TARGETS", id="import-alias"),
    ],
)
def test_guard_tamper_identifiers_are_flagged(code, roots):
    issues = path_policy_issues(code, **roots)
    assert issues, f"no issue raised for: {code}"
    assert "sandbox guard" in issues[0]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("my_osprey_fsx = 1", id="embedded-not-prefixed"),
        pytest.param("osprey_fs = load()", id="no-leading-underscore"),
        pytest.param("x = df._osprey", id="shorter-than-prefix"),
        pytest.param("restore_patched_targets()", id="near-miss-exact-name"),
        pytest.param("_restore_patched_targets_extra = 1", id="exact-name-is-not-a-prefix"),
        pytest.param("'the _osprey_fs guard rejects this path'", id="prose-mentioning-a-name"),
    ],
)
def test_innocent_identifiers_stay_clean(code, roots):
    assert path_policy_issues(code, **roots) == []


def test_guard_tamper_is_flagged_without_any_protected_root():
    """Guard ownership does not depend on the path sets — there is no path."""
    assert path_policy_issues("_restore_patched_targets()", protected_roots=[])


def test_each_distinct_guard_name_is_reported_once():
    code = "_osprey_fs_check(_OSPREY_FS_PROTECTED)\n_osprey_fs_check(1)\n"
    issues = path_policy_issues(code, protected_roots=[])
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


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("import importlib\nimportlib.reload(os)", id="importlib-dotted"),
        pytest.param("from importlib import reload\nreload(shutil)", id="bare-reload"),
        pytest.param("importlib.reload(builtins)", id="builtins"),
        pytest.param("importlib.reload(io)", id="io"),
        pytest.param("importlib.reload(sys.modules['os'])", id="sys-modules-subscript"),
        pytest.param("imp.reload(os)", id="any-receiver"),
    ],
)
def test_reloading_a_guarded_module_is_flagged(code, roots):
    issues = path_policy_issues(code, **roots)
    assert issues, f"no issue raised for: {code}"
    assert "reload" in issues[0]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("import importlib\nimportlib.reload(numpy)", id="unguarded-module"),
        pytest.param("config.reload()", id="method-named-reload"),
        pytest.param("reload()", id="no-argument"),
        pytest.param("os.remove('scratch.txt')", id="ordinary-os-call"),
    ],
)
def test_reloads_that_cannot_disarm_the_guard_stay_clean(code, roots):
    assert path_policy_issues(code, **roots) == []


def test_reload_is_flagged_without_any_protected_root():
    """Same as guard-identifier tampering: it does not depend on the path sets."""
    assert path_policy_issues("importlib.reload(os)", protected_roots=[])


def test_reloading_a_guarded_module_is_flagged__mutation_renames_the_module(roots):
    """Proves the check keys on the guarded module, not on the word 'reload'."""
    with pytest.raises(AssertionError):
        assert path_policy_issues("importlib.reload(os_lookalike)", **roots)

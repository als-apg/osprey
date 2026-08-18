"""Tests for the content fingerprints ``runtime_env`` injects.

The rendered compose templates carry an ``osprey.env.digest`` label
interpolated from ``OSPREY_ENV_DIGEST``, which is what makes an edit to the env
chain recreate the containers that read it. podman-compose does not diff a
running container's environment against the env files it was built from, so
without a label that moves with the chain's contents, ``osprey up`` after an edit
to ``.env`` leaves the old values running.

``OSPREY_CONFIG_DIGEST`` and ``osprey.config.digest`` are the same mechanism for
the other file a container reads settings out of: the rendered ``config.yml``
each service mounts. Neither file is named by the compose document, which is the
only thing compose compares, so both need a label to be seen at all.

Two properties are load-bearing and tested separately here:

* the **recipe** — a single ``sha256`` over the chain files' raw bytes in chain
  order, empty string for an empty chain — because the ``osprey up`` stale-pin
  preflight recomputes it independently and the two must agree; and
* **universality** — every ``runtime_env`` call site gets the value. A call site
  that skipped it would interpolate the label to the empty string, which reads
  to compose as a changed document and recreates the whole stack for nothing.
  Spurious recreates are the failure this task exists to prevent, so the second
  half is enumerated from the source rather than spot-checked.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from osprey.deployment.runtime_helper import (
    CONFIG_DIGEST_VAR,
    ENV_DIGEST_VAR,
    as_built_config_digest,
    env_chain_digest,
    runtime_env,
)

# ---------------------------------------------------------------------------
# The recipe.
# ---------------------------------------------------------------------------


def test_digest_is_empty_for_a_project_with_no_chain_files(tmp_path: Path) -> None:
    """No ``.env.shared`` and no ``.env`` is a valid shape, not an error.

    The label defaults to the empty string (``${OSPREY_ENV_DIGEST:-}``), so an
    empty chain and an unset variable render the same label value.
    """
    assert env_chain_digest(tmp_path) == ""


def test_digest_hashes_the_only_chain_member_present(tmp_path: Path) -> None:
    """A project with just committed defaults still gets a real digest."""
    (tmp_path / ".env.shared").write_text("SHARED=1\n")

    assert env_chain_digest(tmp_path) == hashlib.sha256(b"SHARED=1\n").hexdigest()


def test_digest_concatenates_both_members_in_chain_order(tmp_path: Path) -> None:
    """One sha256 over shared-then-local bytes -- the recipe the preflight repeats."""
    (tmp_path / ".env.shared").write_text("SHARED=1\n")
    (tmp_path / ".env").write_text("LOCAL=2\n")

    assert env_chain_digest(tmp_path) == hashlib.sha256(b"SHARED=1\nLOCAL=2\n").hexdigest()


def test_digest_is_order_sensitive_across_the_chain(tmp_path: Path) -> None:
    """Swapping which file holds which line changes the digest.

    Concatenating in chain order rather than hashing a set means the digest
    distinguishes ``.env.shared`` holding a value from ``.env`` holding it --
    which matters, because only the second one wins.
    """
    (tmp_path / ".env.shared").write_text("A=1\n")
    (tmp_path / ".env").write_text("B=2\n")
    one_way = env_chain_digest(tmp_path)

    (tmp_path / ".env.shared").write_text("B=2\n")
    (tmp_path / ".env").write_text("A=1\n")

    assert env_chain_digest(tmp_path) != one_way


def test_digest_is_stable_across_rewrites_of_identical_bytes(tmp_path: Path) -> None:
    """Nothing outside the bytes is mixed in -- no mtimes, no paths.

    A rebuild that rewrites the chain files unchanged must not recreate a single
    container, so the digest cannot depend on anything a rewrite touches.
    """
    (tmp_path / ".env").write_text("LOCAL=2\n")
    before = env_chain_digest(tmp_path)

    (tmp_path / ".env").write_text("LOCAL=2\n")

    assert env_chain_digest(tmp_path) == before


def test_digest_moves_on_a_comment_only_edit(tmp_path: Path) -> None:
    """Raw bytes, not parsed values: the digest is deliberately over-sensitive.

    Hashing the parsed ``KEY=VALUE`` view would be a second, subtly different
    answer to "did the env change?" than the one every other consumer of these
    files computes. A needless recreate on a comment edit is the cheap side of
    that trade; a missed recreate on a real edit is the expensive one.
    """
    (tmp_path / ".env").write_text("LOCAL=2\n")
    before = env_chain_digest(tmp_path)

    (tmp_path / ".env").write_text("# a note\nLOCAL=2\n")

    assert env_chain_digest(tmp_path) != before


def test_digest_ignores_files_outside_the_named_chain(tmp_path: Path) -> None:
    """Chain membership is a fixed list of names, never a directory scan.

    ``.env.example``, ``.env.users`` and the deploy's own ``build/.env.merged``
    all sit near the chain without being part of it; a digest that swept them in
    would recreate the stack whenever the build regenerated one.
    """
    (tmp_path / ".env").write_text("LOCAL=2\n")
    before = env_chain_digest(tmp_path)

    (tmp_path / ".env.example").write_text("LOCAL=changeme\n")
    (tmp_path / ".env.users").write_text("LOCAL=3\n")

    assert env_chain_digest(tmp_path) == before


def test_runtime_env_carries_the_digest_of_the_configured_repo(tmp_path: Path) -> None:
    """The value in the env is the chain under the repo compose is pinned to."""
    (tmp_path / ".env.shared").write_text("SHARED=1\n")
    (tmp_path / ".env").write_text("LOCAL=2\n")
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}

    env = runtime_env(config, base_env={})

    assert env[ENV_DIGEST_VAR] == hashlib.sha256(b"SHARED=1\nLOCAL=2\n").hexdigest()


def test_runtime_env_digest_tracks_an_edit_to_the_chain(tmp_path: Path) -> None:
    """The whole point: an edit between two ups yields a different label value."""
    (tmp_path / ".env").write_text("LOCAL=2\n")
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}
    before = runtime_env(config, base_env={})[ENV_DIGEST_VAR]

    (tmp_path / ".env").write_text("LOCAL=3\n")

    assert runtime_env(config, base_env={})[ENV_DIGEST_VAR] != before


def test_runtime_env_digest_is_empty_for_a_repo_with_no_chain(tmp_path: Path) -> None:
    """Set-but-empty, not absent: the key is always present so callers can rely on it."""
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}

    env = runtime_env(config, base_env={})

    assert env[ENV_DIGEST_VAR] == ""


def test_runtime_env_recomputes_rather_than_inheriting_a_stale_digest(tmp_path: Path) -> None:
    """A digest already on the base env is overwritten, never trusted.

    ``osprey up`` hands its own environment down to nested invocations (the web
    terminal stack, the post-up hooks), so a value exported by an outer process
    -- or left over in an operator's shell from a previous deploy -- would
    otherwise pin the label to a chain that no longer exists.
    """
    (tmp_path / ".env").write_text("LOCAL=2\n")
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}

    env = runtime_env(config, base_env={ENV_DIGEST_VAR: "stale-from-an-outer-shell"})

    assert env[ENV_DIGEST_VAR] == hashlib.sha256(b"LOCAL=2\n").hexdigest()


def test_runtime_env_digest_does_not_leak_into_the_caller_env(tmp_path: Path) -> None:
    """The existing copy-never-mutate contract covers the new key too."""
    base: dict[str, str] = {}

    runtime_env({"project_root": str(tmp_path)}, base_env=base, ignore_orphans=True)

    assert base == {}


# ---------------------------------------------------------------------------
# The same contract for the as-built config.
#
# A service mounts the rendered config.yml, so a setting an operator changes
# lands in a file the compose document never mentions -- and compose recreates
# on a changed DEFINITION. Everything below is the env-chain recipe applied to
# that file: hash its raw bytes, empty when there is no build yet, and carried by
# every runtime_env call site so the label moves exactly when the config does.
# ---------------------------------------------------------------------------


def _write_as_built(repo_root: Path, text: str) -> Path:
    """Lay down the one as-built config a build leaves behind."""
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    path = build_dir / "config.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_digest_is_empty_before_a_build_has_rendered_one(tmp_path: Path) -> None:
    """Every verb that runs before a build sees no config, which is not an error."""
    assert as_built_config_digest(tmp_path) == ""


def test_config_digest_hashes_the_as_built_config_bytes(tmp_path: Path) -> None:
    """The recipe, so a second reproducer of it cannot disagree."""
    _write_as_built(tmp_path, "connector: virtual_accelerator\n")

    expected = hashlib.sha256(b"connector: virtual_accelerator\n").hexdigest()
    assert as_built_config_digest(tmp_path) == expected


def test_config_digest_is_stable_across_rewrites_of_identical_bytes(tmp_path: Path) -> None:
    """Rebuilding an unchanged deployment must not restart it.

    The property the removed deploy timestamp destroyed: it differed on every
    render, so every ``up`` recreated every container.
    """
    _write_as_built(tmp_path, "connector: virtual_accelerator\n")
    before = as_built_config_digest(tmp_path)

    _write_as_built(tmp_path, "connector: virtual_accelerator\n")

    assert as_built_config_digest(tmp_path) == before


def test_config_digest_moves_when_a_setting_changes(tmp_path: Path) -> None:
    """The whole point, in the shape the operator hits it.

    ``osprey set connector=mock`` rewrites the config and the next ``up`` has to
    recreate the containers that mount it -- otherwise the deployment keeps
    answering with the connector it started on, and the flip silently did
    nothing.
    """
    _write_as_built(tmp_path, "connector: virtual_accelerator\n")
    before = as_built_config_digest(tmp_path)

    _write_as_built(tmp_path, "connector: mock\n")

    assert as_built_config_digest(tmp_path) != before


def test_runtime_env_carries_the_config_digest_of_the_configured_repo(tmp_path: Path) -> None:
    """The value in the env is the build under the repo compose is pinned to."""
    _write_as_built(tmp_path, "connector: mock\n")
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}

    env = runtime_env(config, base_env={})

    assert env[CONFIG_DIGEST_VAR] == hashlib.sha256(b"connector: mock\n").hexdigest()


def test_runtime_env_config_digest_is_empty_without_a_build(tmp_path: Path) -> None:
    """Set-but-empty, like the env digest, so callers can rely on the key."""
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}

    assert runtime_env(config, base_env={})[CONFIG_DIGEST_VAR] == ""


def test_runtime_env_recomputes_rather_than_inheriting_a_stale_config_digest(
    tmp_path: Path,
) -> None:
    """A value left in an outer shell must never pin the label."""
    _write_as_built(tmp_path, "connector: mock\n")
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}

    env = runtime_env(config, base_env={CONFIG_DIGEST_VAR: "stale-from-an-outer-shell"})

    assert env[CONFIG_DIGEST_VAR] == hashlib.sha256(b"connector: mock\n").hexdigest()


def test_the_two_digests_are_independent(tmp_path: Path) -> None:
    """An env-chain edit must not move the config digest, or the reverse.

    They answer different questions about different files, and a deployment that
    edited only one of them should recreate for that reason alone.
    """
    (tmp_path / ".env").write_text("LOCAL=2\n", encoding="utf-8")
    _write_as_built(tmp_path, "connector: mock\n")
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}
    first = runtime_env(config, base_env={})

    (tmp_path / ".env").write_text("LOCAL=3\n", encoding="utf-8")
    second = runtime_env(config, base_env={})

    assert second[ENV_DIGEST_VAR] != first[ENV_DIGEST_VAR]
    assert second[CONFIG_DIGEST_VAR] == first[CONFIG_DIGEST_VAR]


# ---------------------------------------------------------------------------
# Universality: every call site gets the value.
#
# The call sites are enumerated from the source rather than listed by hand, so a
# new one cannot be added without this test exercising its shape. The scan
# covers every `runtime_env(...)` call in the package, which is a superset of
# the compose-invoking ones -- classifying each argv as compose-or-not would be
# fragile, and the extra sites (a `ps` probe, an image inspect) are harmless to
# cover.
# ---------------------------------------------------------------------------


def _runtime_env_call_sites() -> list[tuple[str, str, int, tuple[str, ...]]]:
    """``(module, enclosing function, positional count, keywords)`` per call site."""
    import osprey

    package_root = Path(inspect.getfile(osprey)).parent
    sites: list[tuple[str, str, int, tuple[str, ...]]] = []

    for source_file in sorted(package_root.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), str(source_file))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "runtime_env"):
                continue
            enclosing = "<module>"
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                    enclosing = parent.name
                    break
                parent = parents.get(parent)
            module = source_file.relative_to(package_root.parent).as_posix()
            sites.append(
                (
                    module,
                    enclosing,
                    len(node.args),
                    tuple(sorted(k.arg or "" for k in node.keywords)),
                )
            )
    return sites


#: Call sites the requirement names explicitly, as ``(module, function)``. Every
#: one is a path that can reach ``up`` for the shared compose project, so all of
#: them must interpolate the SAME label value -- one of them resolving its own
#: environment instead would recreate the stack on every deploy.
_REQUIRED_CALL_SITES = {
    ("osprey/deployment/container_lifecycle.py", "_start_stack"),  # deploy up
    ("osprey/deployment/container_lifecycle.py", "_stage_archiver_store"),  # archiver store
    ("osprey/deployment/web_terminals/provision.py", "deploy_up_web_terminals"),  # web
    ("osprey/deployment/web_terminals/lifecycle.py", "decommission_user"),  # roster verbs
    ("osprey/deployment/web_terminals/lifecycle.py", "prune_users"),
}


def test_the_named_call_sites_still_route_through_runtime_env() -> None:
    """Guards the inventory against a call site quietly rolling its own env.

    The digest only prevents spurious recreates while every one of these builds
    its subprocess environment here; one that switched to ``os.environ.copy()``
    would silently drop the label and take the stack down with it.
    """
    found = {(module, function) for module, function, _, _ in _runtime_env_call_sites()}

    assert _REQUIRED_CALL_SITES <= found


@pytest.mark.parametrize("positional", [1, 2])
@pytest.mark.parametrize("ignore_orphans", [False, True])
def test_every_call_shape_in_the_source_receives_the_digest(
    tmp_path: Path, positional: int, ignore_orphans: bool
) -> None:
    """Exercise each argument shape the package actually uses.

    The shapes differ in whether the caller supplies its own ``base_env`` and
    whether it asks for orphan suppression, and the digest must survive all of
    them -- an implementation that only injected the key when it built the env
    from ``os.environ`` would leave every explicit-``base_env`` call site out.
    """
    shapes = {(count, keywords) for _, _, count, keywords in _runtime_env_call_sites()}
    expected_keywords = ("ignore_orphans",) if ignore_orphans else ()
    if (positional, expected_keywords) not in shapes:
        pytest.skip(f"no call site passes {positional} positional args with {expected_keywords}")

    (tmp_path / ".env").write_text("LOCAL=2\n")
    config = {"project_name": "proj-a", "project_root": str(tmp_path)}
    args = [config] if positional == 1 else [config, {}]

    env = runtime_env(*args, ignore_orphans=ignore_orphans)

    assert env[ENV_DIGEST_VAR] == hashlib.sha256(b"LOCAL=2\n").hexdigest()


def test_the_source_scan_actually_finds_call_sites() -> None:
    """The enumeration above is only a guard while it is finding something.

    A rename of ``runtime_env``, or an import style the ``ast.Name`` match does
    not see, would empty the inventory and turn every test above it green for
    the wrong reason.
    """
    assert len(_runtime_env_call_sites()) >= 10
